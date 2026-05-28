import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import detect, guidance, stream

logger = logging.getLogger("uvicorn.error")


app = FastAPI(title="Voice Phishing API Gateway", version="1.0.0")

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(guidance.router)
app.include_router(stream.router)


@app.on_event("startup")
async def _startup() -> None:
    frontend_port = os.getenv("FRONTEND_PORT", "80")
    url = "http://localhost" if frontend_port == "80" else f"http://localhost:{frontend_port}"
    logger.info("=" * 50)
    logger.info(f"  AIShield Demo 접속 주소: {url}")
    logger.info("=" * 50)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

