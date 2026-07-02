from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

load_dotenv()

from database import init_db
from auth.routes import router as auth_router
from routes.stocks import router as stocks_router
from routes.all_routes import (
    portfolio_router, finance_router,
    news_router, alerts_router
)
from scheduler import start_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 FinPilot AI starting up...")
    await init_db()
    start_scheduler()
    print("✅ FinPilot AI ready")
    yield
    # Shutdown
    print("👋 FinPilot AI shutting down...")

app = FastAPI(
    title="FinPilot AI",
    description="Real-time AI trading assistant for NSE stocks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
        "https://finpilot-ai.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router)
app.include_router(stocks_router)
app.include_router(portfolio_router)
app.include_router(finance_router)
app.include_router(news_router)
app.include_router(alerts_router)

@app.get("/")
async def root():
    return {
        "app": "FinPilot AI",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": __import__("datetime").datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)
