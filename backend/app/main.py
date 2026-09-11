import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .engines import Engines
from .ws_routes import router as ws_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convo-ai.backend")

CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model engines (this can take a while on first run) ...")
    Engines.load()
    yield
    logger.info("Shutting down.")


app = FastAPI(title="ConvoAI Backend Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "engines_ready": Engines.ready()}
