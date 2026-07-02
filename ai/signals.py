import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional
from services.market_data import get_history, symbol_to_yf

async def compute_signals(symbol: str, sentiment_score: float = 0.0) -> dict:
    """
    Core AI signal engine.
    Computes technical indicators and generates BUY/SELL/HOLD recommendation
    with confidence score, entry/target/stop-loss, and explanation.
    """
    yf_sym = symbol_to_yf(symbol)
    df = await get_history(yf_sym, period="6mo", interval="1d")

    if df.empty or len(df) < 30:
        return _empty_signal(symbol)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Technical Indicators ─────────────────────────────────────────────────
    rsi_series = ta.rsi(close, length=14)
    rsi = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50.0

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    macd = float(macd_df["MACD_12_26_9"].iloc[-1]) if macd_df is not None else 0.0
    macd_signal = float(macd_df["MACDs_12_26_9"].iloc[-1]) if macd_df is not None else 0.0
    macd_hist = macd - macd_signal

    sma20 = float(ta.sma(close, length=20).iloc[-1])
    sma50 = float(ta.sma(close, length=50).iloc[-1]) if len(close) >= 50 else sma20
    ema20 = float(ta.ema(close, length=20).iloc[-1])

    bb = ta.bbands(close, length=20, std=2)
    bb_upper = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1]) * 1.02
    bb_lower = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1]) * 0.98
    bb_mid = float(bb["BBM_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1])

    stoch = ta.stoch(high, low, close, k=14, d=3)
    stoch_k = float(stoch["STOCHk_14_3_3"].iloc[-1]) if stoch is not None else 50.0

    adx_df = ta.adx(high, low, close, length=14)
    adx = float(adx_df["ADX_14"].iloc[-1]) if adx_df is not None else 20.0

    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_current = float(volume.iloc[-1])
    vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    price_change_pct = ((current_price - prev_price) / prev_price) * 100

    # 5-day momentum
    momentum_5d = float(close.pct_change(5).iloc[-1]) * 100

    # ── Scoring System ────────────────────────────────────────────────────────
    bullish_score = 0
    bearish_score = 0
    reasons = []

    # RSI signals
    if rsi < 30:
        bullish_score += 3
        reasons.append({"ok": True, "text": f"RSI oversold at {rsi:.1f} — potential reversal upward"})
    elif rsi > 70:
        bearish_score += 3
        reasons.append({"ok": False, "text": f"RSI overbought at {rsi:.1f} — caution, possible pullback"})
    elif 40 <= rsi <= 60:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"RSI neutral at {rsi:.1f} — no extreme signal"})
    elif rsi > 60:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"RSI bullish momentum at {rsi:.1f}"})

    # MACD signals
    if macd > macd_signal and macd_hist > 0:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"MACD bullish crossover — upward momentum confirmed"})
    elif macd < macd_signal and macd_hist < 0:
        bearish_score += 2
        reasons.append({"ok": False, "text": f"MACD bearish crossover — downward momentum detected"})

    # Price vs Moving Averages
    if current_price > sma20 and current_price > sma50:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"Price above SMA20 (₹{sma20:.0f}) and SMA50 (₹{sma50:.0f}) — bullish trend"})
    elif current_price < sma20 and current_price < sma50:
        bearish_score += 2
        reasons.append({"ok": False, "text": f"Price below SMA20 (₹{sma20:.0f}) and SMA50 (₹{sma50:.0f}) — bearish trend"})
    elif current_price > sma20:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"Price above SMA20 — short-term bullish"})

    # Bollinger Bands
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    if current_price <= bb_lower * 1.01:
        bullish_score += 2
        reasons.append({"ok": True, "text": "Price near lower Bollinger Band — potential bounce zone"})
    elif current_price >= bb_upper * 0.99:
        bearish_score += 1
        reasons.append({"ok": False, "text": "Price near upper Bollinger Band — resistance zone"})

    # Volume confirmation
    if vol_ratio > 1.5 and price_change_pct > 0:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"High volume ({vol_ratio:.1f}x average) with price rise — strong buying"})
    elif vol_ratio > 1.5 and price_change_pct < 0:
        bearish_score += 2
        reasons.append({"ok": False, "text": f"High volume ({vol_ratio:.1f}x average) with price fall — strong selling"})

    # Momentum
    if momentum_5d > 2:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"Positive 5-day momentum at +{momentum_5d:.1f}%"})
    elif momentum_5d < -2:
        bearish_score += 1
        reasons.append({"ok": False, "text": f"Negative 5-day momentum at {momentum_5d:.1f}%"})

    # Trend strength (ADX)
    if adx > 25:
        reasons.append({"ok": True, "text": f"Strong trend detected (ADX: {adx:.1f}) — trend likely to continue"})

    # Sentiment impact
    if sentiment_score > 0.3:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"Positive news sentiment score: {sentiment_score:+.2f}"})
    elif sentiment_score < -0.3:
        bearish_score += 1
        reasons.append({"ok": False, "text": f"Negative news sentiment score: {sentiment_score:+.2f}"})

    # ── Recommendation ────────────────────────────────────────────────────────
    total = bullish_score + bearish_score
    if total == 0:
        total = 1

    bull_pct = (bullish_score / total) * 100
    bear_pct = (bearish_score / total) * 100

    if bullish_score >= bearish_score + 3:
        recommendation = "BUY"
        confidence = min(95, 50 + (bullish_score - bearish_score) * 5 + (abs(sentiment_score) * 10))
    elif bearish_score >= bullish_score + 3:
        recommendation = "SELL"
        confidence = min(95, 50 + (bearish_score - bullish_score) * 5 + (abs(sentiment_score) * 10))
    else:
        recommendation = "HOLD"
        confidence = min(80, 40 + abs(bullish_score - bearish_score) * 3)

    confidence = round(confidence, 1)
    profit_prob = round(bull_pct * 0.85, 1)   # Conservative adjustment
    loss_prob = round(100 - profit_prob, 1)

    # ── Entry / Target / Stop Loss ────────────────────────────────────────────
    atr = float(ta.atr(high, low, close, length=14).iloc[-1]) if ta.atr(high, low, close, length=14) is not None else current_price * 0.02

    if recommendation == "BUY":
        entry = round(current_price * 0.999, 2)
        target = round(current_price + (atr * 3), 2)
        stop_loss = round(current_price - (atr * 1.5), 2)
    elif recommendation == "SELL":
        entry = round(current_price * 1.001, 2)
        target = round(current_price - (atr * 3), 2)
        stop_loss = round(current_price + (atr * 1.5), 2)
    else:
        entry = round(current_price, 2)
        target = round(current_price + (atr * 2), 2)
        stop_loss = round(current_price - atr, 2)

    reward = abs(target - entry)
    risk_amt = abs(entry - stop_loss)
    rr_ratio = round(reward / risk_amt, 2) if risk_amt > 0 else 1.0
    expected_return_pct = round(((target - entry) / entry) * 100, 1)

    # Risk level
    if confidence > 75 and rr_ratio > 2:
        risk_level = "LOW"
    elif confidence > 60 or rr_ratio > 1.5:
        risk_level = "MED"
    else:
        risk_level = "HIGH"

    # ── Explanation ───────────────────────────────────────────────────────────
    explanation = _generate_explanation(
        symbol, recommendation, confidence, rsi, macd, macd_signal,
        current_price, sma20, sma50, sentiment_score, reasons
    )

    # ── Pattern detection ─────────────────────────────────────────────────────
    pattern = _detect_pattern(close, high, low)

    return {
        "symbol": symbol,
        "recommendation": recommendation,
        "confidence": confidence,
        "profit_probability": profit_prob,
        "loss_probability": loss_prob,
        "entry_price": entry,
        "target_price": target,
        "stop_loss": stop_loss,
        "risk_reward_ratio": rr_ratio,
        "expected_return": f"{'+' if expected_return_pct > 0 else ''}{expected_return_pct}%",
        "risk_level": risk_level,
        "timeframe": "5–10 Days",
        "pattern": pattern,
        "reasons": reasons,
        "explanation": explanation,
        "indicators": {
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "macd_signal": round(macd_signal, 4),
            "sma_20": round(sma20, 2),
            "sma_50": round(sma50, 2),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "volume_ratio": round(vol_ratio, 2),
            "adx": round(adx, 2),
            "stoch_k": round(stoch_k, 2),
        },
        "current_price": current_price,
    }

def _detect_pattern(close, high, low) -> str:
    """Simple pattern detection on recent candles."""
    if len(close) < 20:
        return "Insufficient data"
    recent = close.iloc[-10:]
    if recent.is_monotonic_increasing:
        return "Ascending Channel"
    if recent.is_monotonic_decreasing:
        return "Descending Channel"
    mid = len(recent) // 2
    if recent.iloc[0] < recent.iloc[mid] > recent.iloc[-1]:
        return "Inverted V / Head & Shoulders"
    if recent.iloc[0] > recent.iloc[mid] < recent.iloc[-1]:
        return "V-Shape / Double Bottom"
    if float(close.iloc[-1]) > float(close.rolling(20).max().iloc[-2]):
        return "Breakout"
    return "Consolidation"

def _generate_explanation(symbol, rec, conf, rsi, macd, macd_sig, price, sma20, sma50, sentiment, reasons) -> str:
    pos_reasons = [r["text"] for r in reasons if r["ok"]]
    neg_reasons = [r["text"] for r in reasons if not r["ok"]]

    explanation = f"{symbol} shows a {rec} signal with {conf:.0f}% confidence based on technical analysis. "

    if rec == "BUY":
        explanation += f"The stock is currently trading at ₹{price:.2f}, "
        if price > sma20:
            explanation += f"above its 20-day moving average (₹{sma20:.2f}), indicating positive short-term momentum. "
        if rsi < 50:
            explanation += f"RSI at {rsi:.1f} suggests the stock is not yet overbought, leaving room for upside. "
        if macd > macd_sig:
            explanation += "MACD crossover confirms bullish momentum. "
    elif rec == "SELL":
        explanation += f"The stock at ₹{price:.2f} shows multiple bearish signals. "
        if rsi > 60:
            explanation += f"RSI at {rsi:.1f} indicates overbought conditions. "
        if macd < macd_sig:
            explanation += "MACD crossover confirms bearish momentum. "
    else:
        explanation += f"Mixed signals at ₹{price:.2f} — holding is advised until a clearer trend emerges. "

    if neg_reasons:
        explanation += f"Key risks: {neg_reasons[0]}. "

    explanation += "⚠ All signals are probabilistic estimates. Markets carry risk — always invest within your financial means."
    return explanation

def _empty_signal(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "recommendation": "HOLD",
        "confidence": 0.0,
        "profit_probability": 50.0,
        "loss_probability": 50.0,
        "entry_price": None,
        "target_price": None,
        "stop_loss": None,
        "risk_reward_ratio": None,
        "expected_return": "N/A",
        "risk_level": "HIGH",
        "timeframe": "N/A",
        "pattern": "Insufficient data",
        "reasons": [{"ok": False, "text": "Not enough historical data to generate a signal"}],
        "explanation": "Insufficient data to generate a signal for this stock.",
        "indicators": {},
        "current_price": None,
    }

async def compute_prediction(symbol: str) -> dict:
    """Generate next-day and 5-day probability predictions."""
    yf_sym = symbol_to_yf(symbol)
    df = await get_history(yf_sym, period="1y", interval="1d")

    if df.empty or len(df) < 50:
        return {"symbol": symbol, "bullish": 50, "bearish": 50}

    close = df["Close"]
    returns = close.pct_change().dropna()
    recent_returns = returns.iloc[-20:]

    # Simple statistical probability based on recent momentum
    positive_days = (recent_returns > 0).sum()
    total_days = len(recent_returns)
    bullish_prob = round((positive_days / total_days) * 100, 1)
    bearish_prob = round(100 - bullish_prob, 1)

    # 5-day expected return (mean of recent 5-day windows)
    returns_5d = close.pct_change(5).dropna()
    expected_5d = float(returns_5d.iloc[-5:].mean()) * 100
    std_5d = float(returns_5d.iloc[-20:].std()) * 100

    # Confidence based on trend consistency
    rsi = ta.rsi(close, length=14)
    rsi_val = float(rsi.iloc[-1]) if rsi is not None else 50

    confidence_5d = min(85, max(40, 60 + (bullish_prob - 50) * 0.8))
    risk_5d = min(80, max(20, std_5d * 5))

    short_trend = "Bullish" if bullish_prob > 55 else ("Bearish" if bullish_prob < 45 else "Sideways")
    medium_trend = "Bullish" if float(close.pct_change(20).iloc[-1]) > 0 else "Bearish"

    trend_pred = "BULLISH" if bullish_prob > 55 else ("BEARISH" if bullish_prob < 45 else "SIDEWAYS")
    trend_conf = round(abs(bullish_prob - 50) * 2 + 50, 1)
    strength = round(abs(float(close.pct_change(10).iloc[-1])) * 100, 1)

    return {
        "symbol": symbol,
        "bullish_prob": bullish_prob,
        "bearish_prob": bearish_prob,
        "expected_return_5d": round(expected_5d, 2),
        "confidence_5d": round(confidence_5d, 1),
        "risk_5d": round(risk_5d, 1),
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "trend_prediction": trend_pred,
        "trend_confidence": trend_conf,
        "trend_strength": min(100, strength),
    }
