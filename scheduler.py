from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from database import AsyncSessionLocal
from models import Stock, StockPrice, AISignal, Alert, Prediction, Watchlist, Portfolio
from services.market_data import NSE_STOCKS, get_quote, symbol_to_yf
from services.sentiment import fetch_market_news, compute_overall_sentiment, fetch_stock_news
from ai.signals import compute_signals, compute_prediction
import asyncio
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
scheduler = AsyncIOScheduler(timezone=IST)

async def _get_or_create_stock(db: AsyncSession, symbol: str) -> Stock:
    res = await db.execute(select(Stock).where(Stock.symbol == symbol))
    stock = res.scalar_one_or_none()
    if not stock:
        meta = next((s for s in NSE_STOCKS if s["symbol"] == symbol), None)
        stock = Stock(
            symbol=symbol,
            yf_symbol=f"{symbol}.NS" if not symbol.endswith(".NS") else symbol,
            name=meta["name"] if meta else symbol,
            sector=meta["sector"] if meta else "Unknown"
        )
        db.add(stock)
        await db.flush()
    return stock

async def update_prices():
    """Runs every 1 minute — updates stock prices in DB."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Updating prices...")
    async with AsyncSessionLocal() as db:
        try:
            for s in NSE_STOCKS:
                try:
                    quote = await get_quote(s["yf"])
                    if not quote or not quote.get("price"):
                        continue
                    stock = await _get_or_create_stock(db, s["symbol"])
                    price = StockPrice(
                        stock_id=stock.id,
                        open=quote.get("open", 0),
                        high=quote.get("high", 0),
                        low=quote.get("low", 0),
                        close=quote.get("price", 0),
                        volume=quote.get("volume", 0),
                    )
                    db.add(price)
                    await asyncio.sleep(0.5)  # Avoid rate limiting
                except Exception as e:
                    print(f"Price update error for {s['symbol']}: {e}")
            await db.commit()
            print(f"✅ Prices updated for {len(NSE_STOCKS)} stocks")
        except Exception as e:
            print(f"❌ Price update batch error: {e}")
            await db.rollback()

async def update_signals():
    """Runs every 5 minutes — recomputes AI signals and generates alerts."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🤖 Computing AI signals...")
    async with AsyncSessionLocal() as db:
        for s in NSE_STOCKS[:10]:  # Top 10 stocks
            try:
                symbol = s["symbol"]
                news = await fetch_stock_news(symbol)
                sentiment_score = sum(a.get("sentiment_score", 0) for a in news) / len(news) if news else 0.0
                signal = await compute_signals(symbol, sentiment_score)

                stock = await _get_or_create_stock(db, symbol)
                indicators = signal.get("indicators", {})

                new_sig = AISignal(
                    stock_id=stock.id,
                    recommendation=signal["recommendation"],
                    confidence=signal["confidence"],
                    profit_probability=signal["profit_probability"],
                    loss_probability=signal["loss_probability"],
                    entry_price=signal.get("entry_price"),
                    target_price=signal.get("target_price"),
                    stop_loss=signal.get("stop_loss"),
                    risk_reward_ratio=signal.get("risk_reward_ratio"),
                    expected_return=signal.get("expected_return"),
                    risk_level=signal["risk_level"],
                    timeframe=signal.get("timeframe"),
                    pattern=signal.get("pattern"),
                    reasons=signal.get("reasons", []),
                    explanation=signal.get("explanation"),
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

                # Generate alerts for strong signals
                await _generate_signal_alerts(db, symbol, signal)
                await asyncio.sleep(1)

            except Exception as e:
                print(f"Signal error for {s['symbol']}: {e}")
                await db.rollback()

    print("✅ AI signals updated")

async def update_news():
    """Runs every 15 minutes — fetches latest news."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📰 Fetching news...")
    try:
        from models import NewsArticle
        articles = await fetch_market_news()
        async with AsyncSessionLocal() as db:
            for article in articles:
                try:
                    pub_dt = datetime.fromisoformat(article["published_at"]) if article.get("published_at") else datetime.utcnow()
                    news = NewsArticle(
                        headline=article["headline"],
                        summary=article.get("summary", ""),
                        source=article.get("source", "Unknown"),
                        url=article.get("url", ""),
                        related_symbols=article.get("related_symbols", []),
                        sentiment=article.get("sentiment", "NEUTRAL"),
                        sentiment_score=article.get("sentiment_score", 0.0),
                        published_at=pub_dt,
                    )
                    db.add(news)
                except Exception:
                    pass
            await db.commit()
        print(f"✅ News updated: {len(articles)} articles")
    except Exception as e:
        print(f"❌ News update error: {e}")

async def daily_portfolio_analysis():
    """Runs once daily — portfolio health + risk alerts for all users."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Daily portfolio analysis...")
    async with AsyncSessionLocal() as db:
        try:
            portfolios_res = await db.execute(select(Portfolio))
            holdings = portfolios_res.scalars().all()

            user_holdings = {}
            for h in holdings:
                user_holdings.setdefault(h.user_id, []).append(h)

            for user_id, user_h in user_holdings.items():
                total_invested = sum(h.quantity * h.avg_buy_price for h in user_h)
                if total_invested == 0:
                    continue

                for h in user_h:
                    quote = await get_quote(symbol_to_yf(h.symbol))
                    if not quote:
                        continue
                    current = h.quantity * quote.get("price", h.avg_buy_price)
                    invested = h.quantity * h.avg_buy_price
                    pnl_pct = ((current - invested) / invested) * 100

                    if pnl_pct > 15:
                        alert = Alert(user_id=user_id, alert_type="GOAL", symbol=h.symbol,
                            message=f"{h.symbol} has gained {pnl_pct:.1f}% — consider booking partial profits",
                            severity="MED")
                        db.add(alert)
                    elif pnl_pct < -8:
                        alert = Alert(user_id=user_id, alert_type="RISK", symbol=h.symbol,
                            message=f"{h.symbol} is down {abs(pnl_pct):.1f}% — review stop-loss",
                            severity="HIGH")
                        db.add(alert)

            await db.commit()
            print("✅ Daily portfolio analysis complete")
        except Exception as e:
            print(f"❌ Daily analysis error: {e}")
            await db.rollback()

async def _generate_signal_alerts(db: AsyncSession, symbol: str, signal: dict):
    """Generate alerts for all users watching this stock."""
    rec = signal.get("recommendation")
    conf = signal.get("confidence", 0)

    if rec in ("BUY", "SELL") and conf >= 70:
        # Alert watchlist users
        watch_res = await db.execute(select(Watchlist).where(Watchlist.symbol == symbol))
        watchers = watch_res.scalars().all()
        for w in watchers:
            alert = Alert(
                user_id=w.user_id,
                alert_type="AI",
                symbol=symbol,
                message=f"AI Signal: {rec} {symbol} — {conf:.0f}% confidence. Entry: ₹{signal.get('entry_price', 'N/A')}",
                severity="HIGH" if conf >= 80 else "MED"
            )
            db.add(alert)

def start_scheduler():
    """Register and start all scheduled jobs."""
    # Prices every 1 minute
    scheduler.add_job(update_prices, IntervalTrigger(minutes=1), id="prices", replace_existing=True)
    # AI signals every 5 minutes
    scheduler.add_job(update_signals, IntervalTrigger(minutes=5), id="signals", replace_existing=True)
    # News every 15 minutes
    scheduler.add_job(update_news, IntervalTrigger(minutes=15), id="news", replace_existing=True)
    # Daily analysis at 6 AM IST
    scheduler.add_job(daily_portfolio_analysis, "cron", hour=6, minute=0, id="daily", replace_existing=True)

    scheduler.start()
    print("✅ Scheduler started: prices(1min), signals(5min), news(15min), daily(6AM IST)")
    return scheduler
