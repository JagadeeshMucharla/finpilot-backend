from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel
from typing import Optional, List
from database import get_db
from models import Portfolio, Trade, UserFinance, FinancialGoal, Alert, Watchlist, Stock, AISignal
from auth.jwt import get_current_user
from models import User
from services.market_data import get_quote, symbol_to_yf, get_stock_meta
from services.sentiment import fetch_market_news, compute_overall_sentiment
from datetime import datetime
import asyncio

# ─── PORTFOLIO ROUTER ─────────────────────────────────────────────────────────
portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])

class HoldingAdd(BaseModel):
    symbol: str
    quantity: float
    avg_buy_price: float

class TradeAdd(BaseModel):
    symbol: str
    trade_type: str
    quantity: float
    entry_price: float
    notes: Optional[str] = None

class TradeClose(BaseModel):
    exit_price: float

@portfolio_router.get("/")
async def get_portfolio(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    holdings_res = await db.execute(select(Portfolio).where(Portfolio.user_id == current_user.id))
    holdings = holdings_res.scalars().all()

    result = []
    total_invested = 0
    total_current = 0

    for h in holdings:
        yf_sym = symbol_to_yf(h.symbol)
        quote = await get_quote(yf_sym)
        meta = get_stock_meta(h.symbol)
        price = quote.get("price") or h.current_price or h.avg_buy_price
        invested = h.quantity * h.avg_buy_price
        current = h.quantity * price
        pnl = current - invested
        pnl_pct = (pnl / invested) * 100 if invested else 0

        total_invested += invested
        total_current += current

        result.append({
            "id": h.id,
            "symbol": h.symbol,
            "name": meta.get("name", h.symbol),
            "sector": meta.get("sector", "Unknown"),
            "quantity": h.quantity,
            "avg_buy_price": h.avg_buy_price,
            "current_price": price,
            "change_pct": quote.get("change_pct", 0),
            "invested": round(invested, 2),
            "current_value": round(current, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "added_at": h.added_at.isoformat(),
        })

    total_pnl = total_current - total_invested
    health = _compute_portfolio_health(result)

    return {
        "holdings": result,
        "summary": {
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / total_invested) * 100, 2) if total_invested else 0,
            "positions": len(result),
            "profitable": sum(1 for h in result if h["pnl"] > 0),
        },
        "health": health,
    }

@portfolio_router.post("/add")
async def add_holding(data: HoldingAdd, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    meta = get_stock_meta(data.symbol)
    holding = Portfolio(
        user_id=current_user.id,
        symbol=data.symbol.upper(),
        quantity=data.quantity,
        avg_buy_price=data.avg_buy_price,
        sector=meta.get("sector"),
    )
    db.add(holding)
    await db.commit()
    return {"message": f"Added {data.symbol.upper()} to portfolio", "id": holding.id}

@portfolio_router.delete("/{holding_id}")
async def remove_holding(holding_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Portfolio).where(Portfolio.id == holding_id, Portfolio.user_id == current_user.id))
    holding = res.scalar_one_or_none()
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    await db.delete(holding)
    await db.commit()
    return {"message": "Holding removed"}

@portfolio_router.post("/trade")
async def add_trade(data: TradeAdd, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    meta = get_stock_meta(data.symbol)
    trade = Trade(
        user_id=current_user.id,
        symbol=data.symbol.upper(),
        trade_type=data.trade_type.upper(),
        quantity=data.quantity,
        entry_price=data.entry_price,
        sector=meta.get("sector"),
        notes=data.notes,
        status="OPEN"
    )
    db.add(trade)
    await db.commit()
    return {"message": "Trade added", "id": trade.id}

@portfolio_router.put("/trade/{trade_id}/close")
async def close_trade(trade_id: int, data: TradeClose, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Trade).where(Trade.id == trade_id, Trade.user_id == current_user.id))
    trade = res.scalar_one_or_none()
    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    trade.exit_price = data.exit_price
    trade.close_date = datetime.utcnow()
    trade.status = "CLOSED"

    if trade.trade_type == "BUY":
        trade.pnl = (data.exit_price - trade.entry_price) * trade.quantity
    else:
        trade.pnl = (trade.entry_price - data.exit_price) * trade.quantity

    trade.pnl_pct = ((data.exit_price - trade.entry_price) / trade.entry_price) * 100
    await db.commit()
    return {"message": "Trade closed", "pnl": trade.pnl}

@portfolio_router.get("/trades")
async def get_trades(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Trade).where(Trade.user_id == current_user.id).order_by(desc(Trade.trade_date)))
    trades = res.scalars().all()

    closed = [t for t in trades if t.status == "CLOSED" and t.pnl is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]

    sector_pnl = {}
    for t in closed:
        s = t.sector or "Unknown"
        sector_pnl[s] = sector_pnl.get(s, 0) + (t.pnl or 0)
    best_sector = max(sector_pnl, key=sector_pnl.get) if sector_pnl else "N/A"
    worst_sector = min(sector_pnl, key=sector_pnl.get) if sector_pnl else "N/A"

    return {
        "trades": [{
            "id": t.id, "symbol": t.symbol, "trade_type": t.trade_type,
            "quantity": t.quantity, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
            "status": t.status, "sector": t.sector, "notes": t.notes,
            "trade_date": t.trade_date.isoformat(), "close_date": t.close_date.isoformat() if t.close_date else None,
        } for t in trades],
        "analytics": {
            "total_trades": len(trades),
            "closed_trades": len(closed),
            "win_rate": round((len(wins) / len(closed)) * 100, 1) if closed else 0,
            "total_pnl": round(sum(t.pnl for t in closed), 2),
            "avg_profit": round(sum(t.pnl for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t.pnl for t in losses) / len(losses), 2) if losses else 0,
            "best_sector": best_sector,
            "worst_sector": worst_sector,
        }
    }

@portfolio_router.get("/watchlist")
async def get_watchlist(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Watchlist).where(Watchlist.user_id == current_user.id))
    items = res.scalars().all()
    result = []
    for item in items:
        yf_sym = symbol_to_yf(item.symbol)
        quote = await get_quote(yf_sym)
        meta = get_stock_meta(item.symbol)
        result.append({
            "id": item.id,
            "symbol": item.symbol,
            "name": meta.get("name"),
            "sector": meta.get("sector"),
            "price": quote.get("price"),
            "change_pct": quote.get("change_pct", 0),
            "added_at": item.added_at.isoformat(),
        })
    return {"watchlist": result}

@portfolio_router.post("/watchlist/{symbol}")
async def add_watchlist(symbol: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    item = Watchlist(user_id=current_user.id, symbol=symbol.upper())
    db.add(item)
    await db.commit()
    return {"message": f"{symbol.upper()} added to watchlist"}

@portfolio_router.delete("/watchlist/{item_id}")
async def remove_watchlist(item_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Watchlist).where(Watchlist.id == item_id, Watchlist.user_id == current_user.id))
    item = res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(item)
    await db.commit()
    return {"message": "Removed from watchlist"}

def _compute_portfolio_health(holdings: list) -> dict:
    if not holdings:
        return {"score": 0, "diversification": 0, "risk": 0, "concentration": 0}
    sectors = set(h.get("sector", "Unknown") for h in holdings)
    diversification = min(100, len(sectors) * 20)
    profitable = sum(1 for h in holdings if h["pnl"] > 0)
    win_rate = (profitable / len(holdings)) * 100
    max_allocation = max((h["current_value"] for h in holdings), default=0)
    total = sum(h["current_value"] for h in holdings) or 1
    concentration = 100 - min(100, (max_allocation / total) * 100)
    score = round((diversification * 0.3 + win_rate * 0.4 + concentration * 0.3))
    return {
        "score": score,
        "diversification": round(diversification),
        "win_rate": round(win_rate),
        "concentration": round(concentration),
        "positions": len(holdings),
        "sectors": len(sectors),
    }

# ─── FINANCE ROUTER ───────────────────────────────────────────────────────────
finance_router = APIRouter(prefix="/finance", tags=["finance"])

class FinanceUpdate(BaseModel):
    bank_balance: Optional[float] = None
    monthly_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    emergency_fund: Optional[float] = None

class GoalCreate(BaseModel):
    name: str
    target_amount: float
    saved_amount: float = 0.0
    deadline: str
    icon: str = "🎯"

@finance_router.get("/")
async def get_finance(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(UserFinance).where(UserFinance.user_id == current_user.id))
    f = res.scalar_one_or_none()
    if not f:
        f = UserFinance(user_id=current_user.id, bank_balance=300, monthly_income=6000, monthly_expenses=3000)
        db.add(f)
        await db.commit()

    savings = (f.monthly_income or 0) - (f.monthly_expenses or 0)
    savings_rate = (savings / f.monthly_income * 100) if f.monthly_income else 0
    health_score = _compute_finance_health(f, savings_rate)

    goals_res = await db.execute(select(FinancialGoal).where(FinancialGoal.user_id == current_user.id))
    goals = goals_res.scalars().all()

    return {
        "bank_balance": f.bank_balance,
        "monthly_income": f.monthly_income,
        "monthly_expenses": f.monthly_expenses,
        "monthly_savings": savings,
        "savings_rate": round(savings_rate, 1),
        "emergency_fund": f.emergency_fund,
        "emergency_fund_target": f.emergency_fund_target,
        "investment_budget": f.investment_budget or max(0, savings * 0.5),
        "health_score": health_score,
        "goals": [{
            "id": g.id, "name": g.name, "icon": g.icon,
            "target_amount": g.target_amount, "saved_amount": g.saved_amount,
            "progress_pct": round((g.saved_amount / g.target_amount) * 100, 1) if g.target_amount else 0,
            "deadline": g.deadline,
        } for g in goals],
        "ai_advice": _generate_finance_advice(f, savings, savings_rate),
        "balance_forecast": _forecast_balance(f.bank_balance, savings),
    }

@finance_router.put("/update")
async def update_finance(data: FinanceUpdate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(UserFinance).where(UserFinance.user_id == current_user.id))
    f = res.scalar_one_or_none()
    if not f:
        f = UserFinance(user_id=current_user.id)
        db.add(f)
    for field, value in data.dict(exclude_none=True).items():
        setattr(f, field, value)
    await db.commit()
    return {"message": "Finance updated"}

@finance_router.post("/goals")
async def add_goal(data: GoalCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    goal = FinancialGoal(user_id=current_user.id, **data.dict())
    db.add(goal)
    await db.commit()
    return {"message": "Goal added", "id": goal.id}

@finance_router.put("/goals/{goal_id}/progress")
async def update_goal(goal_id: int, saved_amount: float, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(FinancialGoal).where(FinancialGoal.id == goal_id, FinancialGoal.user_id == current_user.id))
    goal = res.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    goal.saved_amount = saved_amount
    await db.commit()
    return {"message": "Goal updated"}

def _compute_finance_health(f: UserFinance, savings_rate: float) -> dict:
    savings_score = min(100, savings_rate * 2)
    emergency_score = min(100, (f.emergency_fund / f.emergency_fund_target * 100)) if f.emergency_fund_target else 0
    overall = round((savings_score * 0.4 + emergency_score * 0.3 + 40) * 0.9)
    return {
        "score": min(100, overall),
        "savings_rate": round(savings_score),
        "emergency_cover": round(emergency_score),
        "debt_ratio": 92,
    }

def _generate_finance_advice(f: UserFinance, savings: float, savings_rate: float) -> list:
    advice = []
    if savings_rate < 30:
        advice.append({"icon": "◬", "title": "Increase savings rate", "desc": f"Your savings rate is {savings_rate:.0f}%. Aim for at least 30% by reducing discretionary expenses.", "color": "amber"})
    if f.emergency_fund < f.emergency_fund_target:
        needed = f.emergency_fund_target - (f.emergency_fund or 0)
        advice.append({"icon": "🛡️", "title": "Build emergency fund", "desc": f"You need ₹{needed:,.0f} more to reach your emergency fund target. Allocate ₹{needed/12:,.0f}/month.", "color": "amber"})
    if savings > 1000:
        advice.append({"icon": "◆", "title": "Start a SIP", "desc": f"With ₹{savings:,.0f} monthly savings, consider a ₹{savings*0.3:,.0f}/month SIP in a Nifty index fund.", "color": "green"})
    return advice

def _forecast_balance(balance: float, monthly_savings: float) -> list:
    from datetime import date
    forecast = []
    b = balance or 0
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    for i, m in enumerate(months):
        b += monthly_savings
        forecast.append({"month": m, "balance": round(b, 0)})
    return forecast

# ─── NEWS ROUTER ──────────────────────────────────────────────────────────────
news_router = APIRouter(prefix="/news", tags=["news"])

@news_router.get("/")
async def get_news(current_user: User = Depends(get_current_user)):
    articles = await fetch_market_news()
    sentiment = compute_overall_sentiment(articles)
    return {
        "articles": articles,
        "sentiment_summary": sentiment,
        "top_sources": _top_sources(articles),
    }

@news_router.get("/{symbol}")
async def get_stock_news(symbol: str, current_user: User = Depends(get_current_user)):
    from services.sentiment import fetch_stock_news
    articles = await fetch_stock_news(symbol.upper())
    sentiment = compute_overall_sentiment(articles)
    return {"symbol": symbol.upper(), "articles": articles, "sentiment": sentiment}

def _top_sources(articles: list) -> list:
    from collections import Counter
    sources = Counter(a.get("source", "Unknown") for a in articles)
    return [{"source": s, "count": c} for s, c in sources.most_common(5)]

# ─── ALERTS ROUTER ────────────────────────────────────────────────────────────
alerts_router = APIRouter(prefix="/alerts", tags=["alerts"])

@alerts_router.get("/")
async def get_alerts(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(
        select(Alert)
        .where(Alert.user_id == current_user.id)
        .order_by(desc(Alert.created_at))
        .limit(50)
    )
    alerts = res.scalars().all()
    unread = sum(1 for a in alerts if not a.is_read)
    return {
        "alerts": [{
            "id": a.id, "type": a.alert_type, "symbol": a.symbol,
            "message": a.message, "severity": a.severity,
            "is_read": a.is_read, "created_at": a.created_at.isoformat()
        } for a in alerts],
        "unread_count": unread,
    }

@alerts_router.put("/{alert_id}/read")
async def mark_read(alert_id: int, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Alert).where(Alert.id == alert_id, Alert.user_id == current_user.id))
    alert = res.scalar_one_or_none()
    if alert:
        alert.is_read = True
        await db.commit()
    return {"message": "Marked as read"}

@alerts_router.put("/read-all")
async def mark_all_read(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = await db.execute(select(Alert).where(Alert.user_id == current_user.id, Alert.is_read == False))
    alerts = res.scalars().all()
    for a in alerts:
        a.is_read = True
    await db.commit()
    return {"message": f"Marked {len(alerts)} alerts as read"}
