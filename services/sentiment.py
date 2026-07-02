import finnhub
import os
from datetime import datetime, timedelta
from typing import List
import re

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
client = finnhub.Client(api_key=FINNHUB_KEY)

# Sentiment keywords
POSITIVE_WORDS = [
    "growth", "profit", "record", "surge", "rise", "gain", "beat", "strong",
    "positive", "upgrade", "buy", "outperform", "bullish", "expansion", "revenue",
    "milestone", "breakthrough", "success", "rally", "boost", "launch", "win",
    "dividend", "acquisition", "partnership", "deal", "innovation", "up"
]

NEGATIVE_WORDS = [
    "loss", "decline", "fall", "drop", "weak", "miss", "downgrade", "sell",
    "bearish", "risk", "concern", "problem", "cut", "layoff", "lawsuit", "fraud",
    "penalty", "fine", "warning", "recall", "debt", "default", "crash", "down",
    "negative", "disappointing", "underperform", "investigation", "probe"
]

def analyze_sentiment(text: str) -> tuple[float, str]:
    """
    Simple keyword-based sentiment analysis.
    Returns (score, label) where score is -1.0 to +1.0
    """
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0, "NEUTRAL"

    score = (pos_count - neg_count) / total
    score = round(score, 2)

    if score > 0.2:
        return score, "POSITIVE"
    elif score < -0.2:
        return score, "NEGATIVE"
    else:
        return score, "NEUTRAL"

def extract_related_symbols(text: str, symbols: list) -> list:
    """Extract which stock symbols are mentioned in a news article."""
    found = []
    text_upper = text.upper()
    name_map = {
        "RELIANCE": ["RELIANCE", "JIO", "MUKESH AMBANI"],
        "TCS": ["TCS", "TATA CONSULTANCY"],
        "INFY": ["INFOSYS", "INFY"],
        "HDFCBANK": ["HDFC BANK", "HDFCBANK"],
        "ICICIBANK": ["ICICI BANK", "ICICIBANK"],
        "BAJFINANCE": ["BAJAJ FINANCE", "BAJFINANCE"],
        "WIPRO": ["WIPRO"],
        "MARUTI": ["MARUTI", "SUZUKI"],
        "SUNPHARMA": ["SUN PHARMA", "SUNPHARMA"],
        "ITC": ["ITC "],
    }
    for sym, keywords in name_map.items():
        if any(kw in text_upper for kw in keywords):
            found.append(sym)
    return found

async def fetch_market_news() -> List[dict]:
    """Fetch and score general market news."""
    try:
        news_items = client.general_news("general", min_id=0)
        results = []
        for item in news_items[:30]:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            full_text = f"{headline} {summary}"

            score, label = analyze_sentiment(full_text)
            related = extract_related_symbols(full_text, [])
            pub_ts = item.get("datetime", 0)
            pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else datetime.now()

            results.append({
                "headline": headline,
                "summary": summary[:500] if summary else "",
                "source": item.get("source", "Unknown"),
                "url": item.get("url", ""),
                "sentiment": label,
                "sentiment_score": score,
                "related_symbols": related,
                "published_at": pub_dt.isoformat(),
                "time_ago": _time_ago(pub_dt),
            })
        return results
    except Exception as e:
        print(f"News fetch error: {e}")
        return []

async def fetch_stock_news(symbol: str) -> List[dict]:
    """Fetch news for a specific stock."""
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        news_items = client.company_news(symbol, _from=start, to=end)
        results = []
        for item in (news_items or [])[:15]:
            headline = item.get("headline", "")
            summary = item.get("summary", "")
            full_text = f"{headline} {summary}"
            score, label = analyze_sentiment(full_text)
            pub_ts = item.get("datetime", 0)
            pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else datetime.now()
            results.append({
                "headline": headline,
                "summary": summary[:500] if summary else "",
                "source": item.get("source", "Unknown"),
                "url": item.get("url", ""),
                "sentiment": label,
                "sentiment_score": score,
                "published_at": pub_dt.isoformat(),
                "time_ago": _time_ago(pub_dt),
            })
        return results
    except Exception as e:
        print(f"Stock news fetch error for {symbol}: {e}")
        return []

def compute_overall_sentiment(articles: list) -> dict:
    """Compute aggregate sentiment stats across articles."""
    if not articles:
        return {"score": 0.0, "label": "NEUTRAL", "positive": 0, "neutral": 0, "negative": 0, "total": 0}

    pos = sum(1 for a in articles if a.get("sentiment") == "POSITIVE")
    neg = sum(1 for a in articles if a.get("sentiment") == "NEGATIVE")
    neu = sum(1 for a in articles if a.get("sentiment") == "NEUTRAL")
    total = len(articles)

    avg_score = sum(a.get("sentiment_score", 0) for a in articles) / total
    label = "POSITIVE" if avg_score > 0.1 else ("NEGATIVE" if avg_score < -0.1 else "NEUTRAL")

    return {
        "score": round(avg_score, 3),
        "label": label,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "total": total,
        "bullish_pct": round((pos / total) * 100, 1) if total else 0,
    }

def _time_ago(dt: datetime) -> str:
    diff = datetime.now() - dt
    if diff.seconds < 3600:
        return f"{diff.seconds // 60}m ago"
    elif diff.days == 0:
        return f"{diff.seconds // 3600}h ago"
    else:
        return f"{diff.days}d ago"
