"""The two web pages: Ask (chat) and Upload. They only call the JSON API in this application."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
router = APIRouter(include_in_schema=False)


@router.get("/")
def home():
    return RedirectResponse("/chat")


@router.get("/chat")
@router.get("/query/ui")                      # same page under the name used in the course material
def chat_page():
    return FileResponse(WEB_DIR / "chat.html")


@router.get("/upload")
@router.get("/ingest/ui")                     # same page under the name used in the course material
def upload_page():
    return FileResponse(WEB_DIR / "upload.html")
