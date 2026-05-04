"""Page routes — serve HTML templates for each dashboard view."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..app import render_template

router = APIRouter()


@router.get("/")
async def dashboard(request: Request):
    """Dashboard overview page."""
    return render_template("dashboard.html", {"request": request})


@router.get("/memory")
async def memory_page(request: Request):
    """Memory list page."""
    return render_template("memory.html", {"request": request})


@router.get("/graph")
async def graph_page(request: Request):
    """Knowledge graph page."""
    return render_template("graph.html", {"request": request})


@router.get("/timeline")
async def timeline_page(request: Request):
    """Memory timeline page."""
    return render_template("timeline.html", {"request": request})


@router.get("/sessions")
async def sessions_page(request: Request):
    """Sessions page."""
    return render_template("sessions.html", {"request": request})


@router.get("/traces")
async def traces_page(request: Request):
    """Retrieval traces page."""
    return render_template("traces.html", {"request": request})
