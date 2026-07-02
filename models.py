from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum

class RecommendationType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"

class SentimentType(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"

# ─── USERS ────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    finances = relationship("UserFinance", back_populates="user", uselist=False)
    portfolio = relationship("Portfolio", back_populates="user")
    watchlist = relationship("Watchlist", back_populates="user")
    alerts = relationship("Alert", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    goals = relationship("FinancialGoal", back_populates="user")

# ─── PERSONAL FINANCE ─────────────────────────────────────────────────────────
class UserFinance(Base):
    __tablename__ = "user_finances"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    bank_balance = Column(Float, default=300.0)
    monthly_income = Column(Float, default=6000.0)
    monthly_expenses = Column(Float, default=3000.0)
    emergency_fund = Column(Float, default=0.0)
    emergency_fund_target = Column(Float, default=18000.0)
    investment_budget = Column(Float, default=0.0)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    user = relationship("User", back_populates="finances")

class FinancialGoal(Base):
    __tablename__ = "financial_goals"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String(200))
    target_amount = Column(Float)
    saved_amount = Column(Float, default=0.0)
    deadline = Column(String(50))
    icon = Column(String(10), default="🎯")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="goals")

# ─── STOCKS & PRICES ──────────────────────────────────────────────────────────
class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, index=True)   # e.g. RELIANCE
    yf_symbol = Column(String(50))                          # e.g. RELIANCE.NS
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prices = relationship("StockPrice", back_populates="stock")
    signals = relationship("AISignal", back_populates="stock")

class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    stock = relationship("Stock", back_populates="prices")

# ─── AI SIGNALS ───────────────────────────────────────────────────────────────
class AISignal(Base):
    __tablename__ = "ai_signals"
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), index=True)
    recommendation = Column(String(10))          # BUY / SELL / HOLD
    confidence = Column(Float)                    # 0-100
    profit_probability = Column(Float)            # 0-100
    loss_probability = Column(Float)              # 0-100
    entry_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    expected_return = Column(String(20), nullable=True)
    risk_level = Column(String(10))
    timeframe = Column(String(50), nullable=True)
    pattern = Column(String(100), nullable=True)
    reasons = Column(JSON, default=list)
    explanation = Column(Text, nullable=True)
    # Technical indicator values
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    sma_20 = Column(Float, nullable=True)
    ema_50 = Column(Float, nullable=True)
    bb_upper = Column(Float, nullable=True)
    bb_lower = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    stock = relationship("Stock", back_populates="signals")

# ─── PORTFOLIO ────────────────────────────────────────────────────────────────
class Portfolio(Base):
    __tablename__ = "portfolio"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String(50), index=True)
    quantity = Column(Float)
    avg_buy_price = Column(Float)
    current_price = Column(Float, nullable=True)
    sector = Column(String(100), nullable=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    user = relationship("User", back_populates="portfolio")

# ─── WATCHLIST ────────────────────────────────────────────────────────────────
class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String(50))
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="watchlist")

# ─── TRADES ───────────────────────────────────────────────────────────────────
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String(50))
    trade_type = Column(String(10))   # BUY / SELL
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    status = Column(String(20), default="OPEN")   # OPEN / CLOSED
    sector = Column(String(100), nullable=True)
    ai_signal_id = Column(Integer, ForeignKey("ai_signals.id"), nullable=True)
    trade_date = Column(DateTime(timezone=True), server_default=func.now())
    close_date = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    user = relationship("User", back_populates="trades")

# ─── NEWS ─────────────────────────────────────────────────────────────────────
class NewsArticle(Base):
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True)
    headline = Column(Text)
    summary = Column(Text, nullable=True)
    source = Column(String(200))
    url = Column(Text, nullable=True)
    related_symbols = Column(JSON, default=list)
    sentiment = Column(String(20))
    sentiment_score = Column(Float, default=0.0)
    published_at = Column(DateTime(timezone=True))
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())

# ─── ALERTS ───────────────────────────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    alert_type = Column(String(30))   # PRICE / AI / NEWS / TRADE / RISK / GOAL
    symbol = Column(String(50), nullable=True)
    message = Column(Text)
    severity = Column(String(10), default="MED")   # LOW / MED / HIGH
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="alerts")

# ─── PREDICTIONS ──────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), index=True)
    bullish_prob = Column(Float)
    bearish_prob = Column(Float)
    sideways_prob = Column(Float)
    expected_return_5d = Column(Float)
    confidence_5d = Column(Float)
    risk_5d = Column(Float)
    short_trend = Column(String(20))
    medium_trend = Column(String(20))
    trend_prediction = Column(String(20))
    trend_confidence = Column(Float)
    trend_strength = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
