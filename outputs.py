import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    from docx import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

from templates import DocumentTemplate

logger = logging.getLogger(__name__)


@dataclass
class OutputRequest:
    formats: Sequence[str]
    prompt: str
    model: str
    provider: str
    metadata: Dict[str, str]
    include_image: Optional[bytes] = None
    video_bytes: Optional[bytes] = None
    video_format: Optional[str] = None


class OutputManager:
    def __init__(self, base_dir: Path, template: DocumentTemplate):
        self.base_dir = base_dir
        self.template = template

    def create_outputs(self, request: OutputRequest, content: str) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        normalized = [fmt.lower() for fmt in request.formats]
        docx_path = None
        if "docx" in normalized:
            docx_path = self._write_docx(request, content)
            if docx_path:
                paths["docx"] = str(docx_path)
        if "pdf" in normalized:
            pdf_path = self._write_pdf(request, content, docx_path)
            if pdf_path:
                paths["pdf"] = str(pdf_path)
        if "html" in normalized:
            html_path = self._write_html(request, content)
            if html_path:
                paths["html"] = str(html_path)
        if "markdown" in normalized or "md" in normalized:
            md_path = self._write_markdown(request, content)
            if md_path:
                paths["markdown"] = str(md_path)
        if "txt" in normalized:
            txt_path = self._write_plain(request, content)
            if txt_path:
                paths["txt"] = str(txt_path)
        if request.video_bytes and request.video_format:
            video_path = self._write_video(request)
            if video_path:
                paths["video"] = str(video_path)
        if "json" in normalized:
            json_path = self._write_json(request, content, paths)
            if json_path:
                paths["json"] = str(json_path)
        return paths

    def _write_docx(self, request: OutputRequest, content: str) -> Optional[Path]:
        if Document is None:
            logger.error("DOCX export failed: python-docx is not installed")
            return None
        try:
            document = Document()
            metadata = dict(request.metadata)
            metadata.setdefault("footer", f"Generated using {request.provider.upper()}")
            populated = self.template.render_docx(document, content, metadata)
            if request.include_image:
                populated = self.template.add_image(populated, request.include_image, caption="Generated illustration")
            path = self._build_path(request, "docx")
            populated.save(path)
            return path
        except Exception as exc:
            logger.error(f"DOCX export failed: {exc}")
            return None

    def _write_pdf(self, request: OutputRequest, content: str, docx_path: Optional[Path]) -> Optional[Path]:
        target = self._build_path(request, "pdf")
        if docx_path:
            try:
                from docx2pdf import convert
                convert(str(docx_path), str(target))
                return target
            except Exception as exc:
                logger.debug(f"docx2pdf fallback triggered: {exc}")
        try:
            from weasyprint import HTML
            html_string = self.template.render_html(content, request.metadata)
            HTML(string=html_string).write_pdf(str(target))
            return target
        except Exception as exc:
            logger.error(f"PDF export failed: {exc}")
            return None

    def _write_html(self, request: OutputRequest, content: str) -> Optional[Path]:
        try:
            html_string = self.template.render_html(content, request.metadata)
            path = self._build_path(request, "html")
            path.write_text(html_string, encoding="utf-8")
            return path
        except Exception as exc:
            logger.error(f"HTML export failed: {exc}")
            return None

    def _write_markdown(self, request: OutputRequest, content: str) -> Optional[Path]:
        try:
            markdown_string = self.template.render_markdown(content, request.metadata)
            path = self._build_path(request, "md")
            path.write_text(markdown_string, encoding="utf-8")
            return path
        except Exception as exc:
            logger.error(f"Markdown export failed: {exc}")
            return None

    def _write_plain(self, request: OutputRequest, content: str) -> Optional[Path]:
        try:
            plain_string = self.template.render_plain(content, request.metadata)
            path = self._build_path(request, "txt")
            path.write_text(plain_string, encoding="utf-8")
            return path
        except Exception as exc:
            logger.error(f"Text export failed: {exc}")
            return None

    def _write_json(self, request: OutputRequest, content: str, artifacts: Dict[str, str]) -> Optional[Path]:
        try:
            payload = {
                "metadata": request.metadata,
                "content": content,
                "artifacts": artifacts,
            }
            path = self._build_path(request, "json")
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return path
        except Exception as exc:
            logger.error(f"JSON export failed: {exc}")
            return None

    def _write_video(self, request: OutputRequest) -> Optional[Path]:
        try:
            extension = request.video_format.lower().lstrip(".") if request.video_format else "mp4"
            path = self._build_path(request, extension)
            path.write_bytes(request.video_bytes or b"")
            return path
        except Exception as exc:
            logger.error(f"Video export failed: {exc}")
            return None

    def _build_path(self, request: OutputRequest, extension: str) -> Path:
        stem = request.metadata.get("slug", "document")
        path = self.base_dir / f"{stem}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
