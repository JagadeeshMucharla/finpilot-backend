import pandas as pd
import numpy as np
import ta
from services.market_data import get_history, symbol_to_yf

async def compute_signals(symbol: str, sentiment_score: float = 0.0) -> dict:
    yf_sym = symbol_to_yf(symbol)
    df = await get_history(yf_sym, period="6mo", interval="1d")

    if df.empty or len(df) < 30:
        return _empty_signal(symbol)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])

    # MACD
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd = float(macd_ind.macd().iloc[-1])
    macd_signal = float(macd_ind.macd_signal().iloc[-1])
    macd_hist = macd - macd_signal

    # Moving averages
    sma20 = float(ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1])
    sma50 = float(ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]) if len(close) >= 50 else sma20

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])

    # Volume
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_current = float(volume.iloc[-1])
    vol_ratio = vol_current / vol_avg if vol_avg > 0 else 1.0

    current_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    price_change_pct = ((current_price - prev_price) / prev_price) * 100
    momentum_5d = float(close.pct_change(5).iloc[-1]) * 100

    # Scoring
    bullish_score = 0
    bearish_score = 0
    reasons = []

    if rsi < 30:
        bullish_score += 3
        reasons.append({"ok": True, "text": f"RSI oversold at {rsi:.1f} — potential reversal"})
    elif rsi > 70:
        bearish_score += 3
        reasons.append({"ok": False, "text": f"RSI overbought at {rsi:.1f} — possible pullback"})
    elif rsi > 60:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"RSI bullish at {rsi:.1f}"})

    if macd > macd_signal:
        bullish_score += 2
        reasons.append({"ok": True, "text": "MACD bullish crossover confirmed"})
    else:
        bearish_score += 2
        reasons.append({"ok": False, "text": "MACD bearish crossover detected"})

    if current_price > sma20 and current_price > sma50:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"Price above SMA20 and SMA50 — bullish trend"})
    elif current_price < sma20 and current_price < sma50:
        bearish_score += 2
        reasons.append({"ok": False, "text": f"Price below SMA20 and SMA50 — bearish trend"})

    if current_price <= bb_lower * 1.01:
        bullish_score += 2
        reasons.append({"ok": True, "text": "Price near lower Bollinger Band — bounce zone"})
    elif current_price >= bb_upper * 0.99:
        bearish_score += 1
        reasons.append({"ok": False, "text": "Price near upper Bollinger Band — resistance"})

    if vol_ratio > 1.5 and price_change_pct > 0:
        bullish_score += 2
        reasons.append({"ok": True, "text": f"High volume {vol_ratio:.1f}x with price rise"})
    elif vol_ratio > 1.5 and price_change_pct < 0:
        bearish_score += 2
        reasons.append({"ok": False, "text": f"High volume {vol_ratio:.1f}x with price fall"})

    if momentum_5d > 2:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"Positive 5-day momentum +{momentum_5d:.1f}%"})
    elif momentum_5d < -2:
        bearish_score += 1
        reasons.append({"ok": False, "text": f"Negative 5-day momentum {momentum_5d:.1f}%"})

    if sentiment_score > 0.3:
        bullish_score += 1
        reasons.append({"ok": True, "text": f"Positive news sentiment {sentiment_score:+.2f}"})
    elif sentiment_score < -0.3:
        bearish_score += 1
        reasons.append({"ok": False, "text": f"Negative news sentiment {sentiment_score:+.2f}"})

    # Recommendation
    if bullish_score >= bearish_score + 3:
        recommendation = "BUY"
        confidence = min(95, 50 + (bullish_score - bearish_score) * 5)
    elif bearish_score >= bullish_score + 3:
        recommendation = "SELL"
        confidence = min(95, 50 + (bearish_score - bullish_score) * 5)
    else:
        recommendation = "HOLD"
        confidence = min(80, 40 + abs(bullish_score - bearish_score) * 3)

    total = bullish_score + bearish_score or 1
    profit_prob = round((bullish_score / total) * 85, 1)
    loss_prob = round(100 - profit_prob, 1)

    # ATR for entry/target/sl
    atr = float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

    if recommendation == "BUY":
        entry = round(current_price * 0.999, 2)
        target = round(current_price + atr * 3, 2)
        stop_loss = round(current_price - atr * 1.5, 2)
    elif recommendation == "SELL":
        entry = round(current_price * 1.001, 2)
        target = round(current_price - atr * 3, 2)
        stop_loss = round(current_price + atr * 1.5, 2)
    else:
        entry = round(current_price, 2)
        target = round(current_price + atr * 2, 2)
        stop_loss = round(current_price - atr, 2)

    reward = abs(target - entry)
    risk_amt = abs(entry - stop_loss)
    rr_ratio = round(reward / risk_amt, 2) if risk_amt > 0 else 1.0
    expected_return_pct = round(((target - entry) / entry) * 100, 1)

    risk_level = "LOW" if confidence > 75 and rr_ratio > 2 else ("HIGH" if confidence < 60 else "MED")

    explanation = f"{symbol} shows a {recommendation} signal with {confidence:.0f}% confidence. "
    explanation += f"Key factors: {reasons[0]['text'] if reasons else 'Mixed signals'}. "
    explanation += "⚠ All signals are probabilistic. Always invest within your means."

    pattern = _detect_pattern(close)

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
        "timeframe": "5-10 Days",
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
        },
        "current_price": current_price,
    }

def _detect_pattern(close) -> str:
    if len(close) < 20:
        return "Insufficient data"
    recent = close.iloc[-10:]
    if recent.is_monotonic_increasing:
        return "Ascending Channel"
    if recent.is_monotonic_decreasing:
        return "Descending Channel"
    if float(close.iloc[-1]) > float(close.rolling(20).max().iloc[-2]):
        return "Breakout"
    return "Consolidation"

async def compute_prediction(symbol: str) -> dict:
    yf_sym = symbol_to_yf(symbol)
    df = await get_history(yf_sym, period="1y", interval="1d")

    if df.empty or len(df) < 50:
        return {"symbol": symbol, "bullish_prob": 50, "bearish_prob": 50}

    close = df["Close"]
    returns = close.pct_change().dropna()
    recent = returns.iloc[-20:]
    bullish_prob = round((recent > 0).sum() / len(recent) * 100, 1)
    bearish_prob = round(100 - bullish_prob, 1)
    expected_5d = round(float(close.pct_change(5).iloc[-5:].mean()) * 100, 2)
    std_5d = round(float(returns.iloc[-20:].std()) * 100 * 5, 1)
    confidence_5d = round(min(85, max(40, 60 + (bullish_prob - 50) * 0.8)), 1)
    short_trend = "Bullish" if bullish_prob > 55 else ("Bearish" if bullish_prob < 45 else "Sideways")
    medium_trend = "Bullish" if float(close.pct_change(20).iloc[-1]) > 0 else "Bearish"
    trend_pred = "BULLISH" if bullish_prob > 55 else ("BEARISH" if bullish_prob < 45 else "SIDEWAYS")

    return {
        "symbol": symbol,
        "bullish_prob": bullish_prob,
        "bearish_prob": bearish_prob,
        "expected_return_5d": expected_5d,
        "confidence_5d": confidence_5d,
        "risk_5d": round(std_5d, 1),
        "short_trend": short_trend,
        "medium_trend": medium_trend,
        "trend_prediction": trend_pred,
        "trend_confidence": round(abs(bullish_prob - 50) * 2 + 50, 1),
        "trend_strength": min(100, round(abs(float(close.pct_change(10).iloc[-1])) * 100, 1)),
    }

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
        "reasons": [{"ok": False, "text": "Not enough data"}],
        "explanation": "Insufficient data to generate signal.",
        "indicators": {},
        "current_price": None,
    }
