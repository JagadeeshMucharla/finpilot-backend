from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="FinPilot AI",
    description="Real-time AI trading assistant for NSE stocks",
    version="1.0.0"
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
        "https://finpilot-frontend.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    print("FinPilot AI starting...")
    from database import init_db
    await init_db()
    from scheduler import start_scheduler
    start_scheduler()
    print("FinPilot AI ready!")

@app.on_event("shutdown")
async def shutdown():
    print("FinPilot AI shutting down...")

from auth.routes import router as auth_router
from routes.stocks import router as stocks_router
from routes.all_routes import (
    portfolio_router, finance_router,
    news_router, alerts_router
)

app.include_router(auth_router)
app.include_router(stocks_router)
app.include_router(portfolio_router)
app.include_router(finance_router)
app.include_router(news_router)
app.include_router(alerts_router)

@app.get("/")
async def root():
    return {"app": "FinPilot AI", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
