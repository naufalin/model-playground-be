"""FastAPI application with lifespan management."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from playground.auth.router import router as auth_router  # noqa: E402
from playground.db.repos.model_repo import ModelRepo  # noqa: E402
from playground.deps import get_db  # noqa: E402
from playground.models.router import router as models_router  # noqa: E402
from playground.playground.router import router as playground_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect DB, verify model registry
    db = await get_db()
    model_repo = ModelRepo(db)
    models = await model_repo.list_active()
    if not models:
        import logging

        logging.warning("No active models in registry — run alembic upgrade head first")
    yield
    # Shutdown: disconnect DB
    await db.disconnect()


app = FastAPI(
    title="LLM Playground",
    description="Test multiple LLMs side-by-side",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(models_router)
app.include_router(playground_router)


@app.get("/")
async def health():
    return {"status": "ok", "version": "0.1.0"}
