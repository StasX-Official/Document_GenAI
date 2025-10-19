"""FastAPI web interface for the AI Document Generator."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import Config, SupportedFormat
from generator import DocumentGenerator

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "web_templates"
STATIC_DIR = BASE_DIR / "web_static"

TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Document Generator", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

generator = DocumentGenerator()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    models = Config.list_models()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": models,
            "default_model": generator.config.settings.get("default_model"),
            "default_formats": generator.config.settings.get("default_formats", [SupportedFormat.DOCX.value]),
        },
    )


@app.post("/api/generate")
async def generate(
    request: Request,
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    formats: Optional[str] = Form(None),
    enable_editing: Optional[str] = Form("false"),
) -> JSONResponse:
    formats_list = [fmt.strip() for fmt in (formats or "").split(",") if fmt.strip()]
    enable_editing_flag = str(enable_editing).lower() in {"true", "1", "on", "yes"}
    try:
        result = await generator.generate_bundle(
            prompt,
            model or generator.config.settings.get("default_model"),
            formats=formats_list,
            enable_editing=enable_editing_flag,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate content")

    return JSONResponse(
        {
            "content": result.content,
            "files": result.files,
            "model": result.model,
            "prompt": result.prompt,
        }
    )


@app.get("/api/history")
async def history(limit: int = 10) -> Dict[str, List[Dict[str, str]]]:
    entries = generator.config.get_recent_history(limit)
    return {"history": [entry.__dict__ for entry in entries]}


def start(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    start()
