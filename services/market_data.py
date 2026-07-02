import yfinance as yf
import finnhub
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)
executor = ThreadPoolExecutor(max_workers=4)

# Top NSE stocks tracked by FinPilot
NSE_STOCKS = [
    {"symbol": "RELIANCE",    "yf": "RELIANCE.NS",    "sector": "Energy",            "name": "Reliance Industries"},
    {"symbol": "TCS",         "yf": "TCS.NS",          "sector": "IT",                "name": "Tata Consultancy Services"},
    {"symbol": "INFY",        "yf": "INFY.NS",         "sector": "IT",                "name": "Infosys"},
    {"symbol": "HDFCBANK",    "yf": "HDFCBANK.NS",     "sector": "Finance",           "name": "HDFC Bank"},
    {"symbol": "ICICIBANK",   "yf": "ICICIBANK.NS",    "sector": "Finance",           "name": "ICICI Bank"},
    {"symbol": "BAJFINANCE",  "yf": "BAJFINANCE.NS",   "sector": "Finance",           "name": "Bajaj Finance"},
    {"symbol": "WIPRO",       "yf": "WIPRO.NS",        "sector": "IT",                "name": "Wipro"},
    {"symbol": "MARUTI",      "yf": "MARUTI.NS",       "sector": "Automobile",        "name": "Maruti Suzuki"},
    {"symbol": "SUNPHARMA",   "yf": "SUNPHARMA.NS",    "sector": "Healthcare",        "name": "Sun Pharma"},
    {"symbol": "ITC",         "yf": "ITC.NS",          "sector": "Consumer Goods",    "name": "ITC"},
    {"symbol": "AXISBANK",    "yf": "AXISBANK.NS",     "sector": "Finance",           "name": "Axis Bank"},
    {"symbol": "KOTAKBANK",   "yf": "KOTAKBANK.NS",    "sector": "Finance",           "name": "Kotak Mahindra Bank"},
    {"symbol": "LT",          "yf": "LT.NS",           "sector": "Infrastructure",    "name": "Larsen & Toubro"},
    {"symbol": "ASIANPAINT",  "yf": "ASIANPAINT.NS",   "sector": "Consumer Goods",    "name": "Asian Paints"},
    {"symbol": "TITAN",       "yf": "TITAN.NS",        "sector": "Consumer Goods",    "name": "Titan Company"},
    {"symbol": "NESTLEIND",   "yf": "NESTLEIND.NS",    "sector": "Consumer Goods",    "name": "Nestle India"},
    {"symbol": "HCLTECH",     "yf": "HCLTECH.NS",      "sector": "IT",                "name": "HCL Technologies"},
    {"symbol": "TATAMOTORS",  "yf": "TATAMOTORS.NS",   "sector": "Automobile",        "name": "Tata Motors"},
    {"symbol": "NTPC",        "yf": "NTPC.NS",         "sector": "Energy",            "name": "NTPC"},
    {"symbol": "ONGC",        "yf": "ONGC.NS",         "sector": "Energy",            "name": "ONGC"},
]

def _fetch_quote_sync(yf_symbol: str) -> dict:
    """Fetch current quote synchronously (runs in thread pool)."""
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        hist = ticker.history(period="2d", interval="1d")
        if hist.empty:
            return {}
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
        close = float(latest["Close"])
        prev_close = float(prev["Close"])
        change = close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        return {
            "price": round(close, 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        print(f"Error fetching {yf_symbol}: {e}")
        return {}

def _fetch_history_sync(yf_symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history synchronously."""
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching history for {yf_symbol}: {e}")
        return pd.DataFrame()

def _fetch_info_sync(yf_symbol: str) -> dict:
    """Fetch company fundamentals synchronously."""
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "book_value": info.get("bookValue"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": info.get("longBusinessSummary", ""),
            "employees": info.get("fullTimeEmployees"),
        }
    except Exception as e:
        print(f"Error fetching info for {yf_symbol}: {e}")
        return {}

async def get_quote(yf_symbol: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _fetch_quote_sync, yf_symbol)

async def get_history(yf_symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _fetch_history_sync, yf_symbol, period, interval)

async def get_company_info(yf_symbol: str) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _fetch_info_sync, yf_symbol)

async def get_all_quotes() -> list:
    """Fetch quotes for all tracked stocks concurrently."""
    tasks = [get_quote(s["yf"]) for s in NSE_STOCKS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for i, result in enumerate(results):
        if isinstance(result, dict) and result:
            stock = NSE_STOCKS[i].copy()
            stock.update(result)
            output.append(stock)
    return output

def get_finnhub_news(symbol: str) -> list:
    """Fetch company news from Finnhub."""
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        # Map NSE symbol to Finnhub format (US-listed ADRs or general search)
        news = finnhub_client.company_news(symbol, _from=start, to=end)
        return news[:10] if news else []
    except Exception as e:
        print(f"Finnhub news error for {symbol}: {e}")
        return []

def get_market_news() -> list:
    """Fetch general market news."""
    try:
        news = finnhub_client.general_news("general", min_id=0)
        return news[:20] if news else []
    except Exception as e:
        print(f"Finnhub market news error: {e}")
        return []

def symbol_to_yf(symbol: str) -> str:
    """Convert NSE symbol to yfinance format."""
    for s in NSE_STOCKS:
        if s["symbol"] == symbol.upper():
            return s["yf"]
    return f"{symbol.upper()}.NS"

def get_stock_meta(symbol: str) -> dict:
    """Get stock metadata from our list."""
    for s in NSE_STOCKS:
        if s["symbol"] == symbol.upper():
            return s
    return {"symbol": symbol.upper(), "yf": f"{symbol.upper()}.NS", "sector": "Unknown", "name": symbol}
