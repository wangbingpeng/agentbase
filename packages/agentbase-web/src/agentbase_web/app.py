"""AgentBase Web Dashboard — FastAPI application factory and lifecycle."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from agentbase import AgentBase
from agentbase_core.models.config import AgentBaseConfig

logger = logging.getLogger(__name__)

# Global AgentBase instance per process
_db: AgentBase | None = None

# Resolve paths relative to this file
_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _DIR / "templates"
_STATIC_DIR = _DIR / "static"

# Direct Jinja2 Environment — bypasses starlette Jinja2Templates cache_key bug
# (jinja2 3.1.6 + starlette 1.0.0: unhashable type 'dict' in cache)
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    auto_reload=False,
    autoescape=True,
)


def render_template(name: str, context: dict) -> HTMLResponse:
    """Render a Jinja2 template and return HTMLResponse."""
    template = _jinja_env.get_template(name)
    return HTMLResponse(template.render(context))


def get_db() -> AgentBase:
    """Get the current AgentBase instance."""
    if _db is None:
        raise RuntimeError("AgentBase not initialized")
    return _db


def create_app(db_path: str | Path = "agentbase.db") -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        global _db
        # Enable all features for the dashboard visualization
        config = AgentBaseConfig(
            data_dir=Path(db_path).parent,
            db_filename=Path(db_path).name,
        )
        config.graph.enabled = True
        config.session.enabled = True
        config.observability.enabled = True
        _db = AgentBase(config=config)
        await _db.initialize()
        logger.info(f"AgentBase Web initialized: {db_path}")
        yield
        await _db.close()
        _db = None
        logger.info("AgentBase Web shutdown")

    app = FastAPI(
        title="AgentBase Dashboard",
        description="Context & Memory Visualization Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Register API routers
    from .api import analytics, context, graph, metrics, session, trace

    app.include_router(analytics.router, prefix="/api", tags=["analytics"])
    app.include_router(metrics.router, prefix="/api", tags=["metrics"])
    app.include_router(context.router, prefix="/api", tags=["context"])
    app.include_router(graph.router, prefix="/api", tags=["graph"])
    app.include_router(session.router, prefix="/api", tags=["session"])
    app.include_router(trace.router, prefix="/api", tags=["trace"])

    # Register page routes
    from .api.pages import router as pages_router

    app.include_router(pages_router)

    return app


def main() -> None:
    """Entry point for `agentbase-web` CLI."""
    import sys
    import uvicorn

    db_path = sys.argv[1] if len(sys.argv) > 1 else "agentbase.db"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    uvicorn.run(
        "agentbase_web.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=False,
    )
