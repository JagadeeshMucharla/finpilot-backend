from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from database import get_db
from models import Stock, StockPrice, AISignal, Prediction
from auth.jwt import get_current_user
from models import User
from services.market_data import (
    get_quote, get_history, get_company_info,
    get_all_quotes, symbol_to_yf, get_stock_meta, NSE_STOCKS
)
from services.sentiment import fetch_stock_news, analyze_sentiment
from ai.signals import compute_signals, compute_prediction
from datetime import datetime
import asyncio

router = APIRouter(prefix="/stocks", tags=["stocks"])

@router.get("/screener")
async def screener(
    sector: str = Query(None),
    search: str = Query(None),
    sort_by: str = Query("confidence"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Market screener — all stocks with live prices and AI signals."""
    stocks = NSE_STOCKS.copy()

    # Filter
    if sector:
        stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]
    if search:
        stocks = [s for s in stocks if search.upper() in s["symbol"] or search.lower() in s["name"].lower()]

    results = []
    # Fetch latest signal from DB for each stock
    for s in stocks[:20]:  # Limit to avoid timeout
        sym = s["symbol"]
        sig_result = await db.execute(
            select(AISignal)
            .join(Stock, Stock.id == AISignal.stock_id)
            .where(Stock.symbol == sym)
            .order_by(desc(AISignal.created_at))
            .limit(1)
        )
        sig = sig_result.scalar_one_or_none()

        quote_result = await db.execute(
            select(StockPrice)
            .join(Stock, Stock.id == StockPrice.stock_id)
            .where(Stock.symbol == sym)
            .order_by(desc(StockPrice.timestamp))
            .limit(2)
        )
        prices = quote_result.scalars().all()

        price = prices[0].close if prices else None
        prev_close = prices[1].close if len(prices) > 1 else price
        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if price and prev_close else 0

        results.append({
            "symbol": sym,
            "name": s["name"],
            "sector": s["sector"],
            "price": price,
            "change_pct": change_pct,
            "recommendation": sig.recommendation if sig else "HOLD",
            "confidence": sig.confidence if sig else 0,
            "profit_probability": sig.profit_probability if sig else 50,
            "risk_level": sig.risk_level if sig else "MED",
            "sentiment_score": sig.sentiment_score if sig else 0,
            "expected_return": sig.expected_return if sig else "N/A",
            "entry_price": sig.entry_price if sig else None,
            "target_price": sig.target_price if sig else None,
            "stop_loss": sig.stop_loss if sig else None,
            "last_analysis": sig.created_at.isoformat() if sig else None,
            "suggested_investment": _suggest_investment(sig.confidence if sig else 0, sig.risk_level if sig else "HIGH"),
        })

    # Sort
    sort_map = {
        "confidence": lambda x: -(x.get("confidence") or 0),
        "profit": lambda x: -(x.get("profit_probability") or 0),
        "risk": lambda x: {"LOW": 0, "MED": 1, "HIGH": 2}.get(x.get("risk_level", "HIGH"), 2),
    }
    if sort_by in sort_map:
        results.sort(key=sort_map[sort_by])

    return {"stocks": results, "total": len(results)}

@router.get("/quote/{symbol}")
async def get_stock_quote(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Get live quote for a stock."""
    yf_sym = symbol_to_yf(symbol)
    quote = await get_quote(yf_sym)
    meta = get_stock_meta(symbol)
    return {**meta, **quote, "symbol": symbol.upper()}

@router.get("/signal/{symbol}")
async def get_signal(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get latest AI signal for a stock. Computes fresh if not cached."""
    symbol = symbol.upper()

    # Try DB first
    stock_res = await db.execute(select(Stock).where(Stock.symbol == symbol))
    stock = stock_res.scalar_one_or_none()

    if stock:
        sig_res = await db.execute(
            select(AISignal)
            .where(AISignal.stock_id == stock.id)
            .order_by(desc(AISignal.created_at))
            .limit(1)
        )
        sig = sig_res.scalar_one_or_none()
        if sig:
            from datetime import timezone
            age_mins = (datetime.now() - sig.created_at.replace(tzinfo=None)).total_seconds() / 60
            if age_mins < 10:  # Return cached if < 10 min old
                return _signal_response(sig, symbol)

    # Compute fresh signal
    news = await fetch_stock_news(symbol)
    sentiment_score = sum(a.get("sentiment_score", 0) for a in news) / len(news) if news else 0.0
    signal_data = await compute_signals(symbol, sentiment_score)

    # Save to DB
    if not stock:
        meta = get_stock_meta(symbol)
        stock = Stock(symbol=symbol, yf_symbol=meta["yf"], name=meta["name"], sector=meta["sector"])
        db.add(stock)
        await db.flush()

    indicators = signal_data.get("indicators", {})
    new_sig = AISignal(
        stock_id=stock.id,
        recommendation=signal_data["recommendation"],
        confidence=signal_data["confidence"],
        profit_probability=signal_data["profit_probability"],
        loss_probability=signal_data["loss_probability"],
        entry_price=signal_data["entry_price"],
        target_price=signal_data["target_price"],
        stop_loss=signal_data["stop_loss"],
        risk_reward_ratio=signal_data["risk_reward_ratio"],
        expected_return=signal_data["expected_return"],
        risk_level=signal_data["risk_level"],
        timeframe=signal_data["timeframe"],
        pattern=signal_data["pattern"],
        reasons=signal_data["reasons"],
        explanation=signal_data["explanation"],
        rsi=indicators.get("rsi"),
        macd=indicators.get("macd"),
        macd_signal=indicators.get("macd_signal"),
        sma_20=indicators.get("sma_20"),
        ema_50=indicators.get("sma_50"),
        bb_upper=indicators.get("bb_upper"),
        bb_lower=indicators.get("bb_lower"),
        volume_ratio=indicators.get("volume_ratio"),
        sentiment_score=sentiment_score,
    )
    db.add(new_sig)
    await db.commit()

    return signal_data

@router.get("/signals/all")
async def get_all_signals(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get latest AI signals for all tracked stocks."""
    results = []
    for s in NSE_STOCKS[:10]:
        try:
            sym = s["symbol"]
            stock_res = await db.execute(select(Stock).where(Stock.symbol == sym))
            stock = stock_res.scalar_one_or_none()
            if stock:
                sig_res = await db.execute(
                    select(AISignal)
                    .where(AISignal.stock_id == stock.id)
                    .order_by(desc(AISignal.created_at))
                    .limit(1)
                )
                sig = sig_res.scalar_one_or_none()
                if sig:
                    results.append(_signal_response(sig, sym))
        except Exception as e:
            print(f"Signal error for {s['symbol']}: {e}")
    return {"signals": results}

@router.get("/history/{symbol}")
async def get_stock_history(
    symbol: str,
    period: str = Query("6mo"),
    interval: str = Query("1d"),
    current_user: User = Depends(get_current_user)
):
    """Get OHLCV history for charting."""
    yf_sym = symbol_to_yf(symbol)
    df = await get_history(yf_sym, period=period, interval=interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "time": str(idx.date()),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })
    return {"symbol": symbol, "period": period, "interval": interval, "candles": candles}

@router.get("/research/{symbol}")
async def get_company_research(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Full company research - fundamentals, news, historical."""
    yf_sym = symbol_to_yf(symbol)
    meta = get_stock_meta(symbol)

    fundamentals, news, history = await asyncio.gather(
        get_company_info(yf_sym),
        fetch_stock_news(symbol),
        get_history(yf_sym, period="5y", interval="1mo"),
    )

    yearly_perf = []
    if not history.empty:
        for year in range(2020, datetime.now().year + 1):
            year_data = history[history.index.year == year]
            if not year_data.empty:
                start = float(year_data["Close"].iloc[0])
                end = float(year_data["Close"].iloc[-1])
                ret = round(((end - start) / start) * 100, 1)
                yearly_perf.append({"year": str(year), "return": f"{'+' if ret > 0 else ''}{ret}%"})

    return {
        "symbol": symbol,
        "name": meta.get("name"),
        "sector": meta.get("sector"),
        "fundamentals": fundamentals,
        "news": news[:10],
        "historical_performance": yearly_perf,
    }

@router.get("/predict/{symbol}")
async def get_prediction(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get AI probability predictions for a stock."""
    prediction = await compute_prediction(symbol.upper())
    return prediction

@router.get("/predict/all")
async def get_all_predictions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get predictions for all tracked stocks."""
    tasks = [compute_prediction(s["symbol"]) for s in NSE_STOCKS[:8]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {"predictions": [r for r in results if isinstance(r, dict)]}

@router.get("/sectors")
async def get_sectors(current_user: User = Depends(get_current_user)):
    """Get list of all sectors."""
    sectors = list(set(s["sector"] for s in NSE_STOCKS))
    return {"sectors": sorted(sectors)}

def _signal_response(sig: AISignal, symbol: str) -> dict:
    return {
        "symbol": symbol,
        "recommendation": sig.recommendation,
        "confidence": sig.confidence,
        "profit_probability": sig.profit_probability,
        "loss_probability": sig.loss_probability,
        "entry_price": sig.entry_price,
        "target_price": sig.target_price,
        "stop_loss": sig.stop_loss,
        "risk_reward_ratio": sig.risk_reward_ratio,
        "expected_return": sig.expected_return,
        "risk_level": sig.risk_level,
        "timeframe": sig.timeframe,
        "pattern": sig.pattern,
        "reasons": sig.reasons or [],
        "explanation": sig.explanation,
        "indicators": {
            "rsi": sig.rsi,
            "macd": sig.macd,
            "macd_signal": sig.macd_signal,
            "sma_20": sig.sma_20,
            "ema_50": sig.ema_50,
            "bb_upper": sig.bb_upper,
            "bb_lower": sig.bb_lower,
            "volume_ratio": sig.volume_ratio,
        },
        "last_updated": sig.created_at.isoformat() if sig.created_at else None,
    }

def _suggest_investment(confidence: float, risk_level: str) -> int:
    """Suggest investment amount based on user's ₹3,000 monthly savings."""
    monthly_savings = 3000
    if risk_level == "LOW" and confidence > 70:
        return int(monthly_savings * 0.4)    # ₹1,200
    elif risk_level == "MED" and confidence > 60:
        return int(monthly_savings * 0.25)   # ₹750
    elif risk_level == "HIGH":
        return int(monthly_savings * 0.1)    # ₹300
    return 0
