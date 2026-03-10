import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import detect, guidance, stream


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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

