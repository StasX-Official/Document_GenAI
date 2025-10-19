"""Configuration handling for the AI Document Generator."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class ProviderType(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    XAI = "xai"
    DIFFUSERS = "diffusers"


class ModelType(str, Enum):
    GEMINI_PRO = "gemini:gemini-1.5-pro"
    GEMINI_FLASH = "gemini:gemini-1.5-flash"
    GEMINI_EXPRESS = "gemini:gemini-1.5-flash-8b"
    GPT_4O = "openai:gpt-4o"
    GPT_4O_MINI = "openai:gpt-4o-mini"
    GPT_4O_AUDIO = "openai:gpt-4o-audio-preview"
    GPT_4_1 = "openai:gpt-4.1"
    GPT_O1 = "openai:o1"
    GROK_3 = "xai:grok-3"
    GROK_3_MINI = "xai:grok-3-mini"
    DIFFUSERS_LOCAL = "diffusers:stable-diffusion-v1-5"


MODEL_REGISTRY: Dict[str, List[str]] = {
    ProviderType.GEMINI.value: [
        ModelType.GEMINI_PRO.value,
        ModelType.GEMINI_FLASH.value,
        ModelType.GEMINI_EXPRESS.value,
    ],
    ProviderType.OPENAI.value: [
        ModelType.GPT_4O.value,
        ModelType.GPT_4O_MINI.value,
        ModelType.GPT_4O_AUDIO.value,
        ModelType.GPT_4_1.value,
        ModelType.GPT_O1.value,
    ],
    ProviderType.XAI.value: [
        ModelType.GROK_3.value,
        ModelType.GROK_3_MINI.value,
    ],
    ProviderType.DIFFUSERS.value: [ModelType.DIFFUSERS_LOCAL.value],
}


class SupportedFormat(str, Enum):
    DOCX = "docx"
    PDF = "pdf"
    PPTX = "pptx"
    HTML = "html"
    MARKDOWN = "md"
    TEXT = "txt"


@dataclass
class HistoryEntry:
    timestamp: str
    prompt: str
    model: str
    filepath: str
    formats: List[str]


class Config:
    """Persisted configuration and history utilities."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "api_keys": {
            "gemini": "",
            "openai": "",
            "xai": "",
        },
        "default_model": ModelType.GEMINI_FLASH.value,
        "output_directory": "generated_documents",
        "document_template": "default",
        "log_level": "INFO",
        "history_enabled": True,
        "max_history_items": 100,
        "default_temperature": 0.6,
        "auto_open_editor": True,
        "default_formats": [SupportedFormat.DOCX.value],
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_dir = Path.home() / ".ai_docs"
        self.config_path = config_path or self.config_dir / "config.json"
        self.history_path = self.config_dir / "history.json"
        self.revision_path = self.config_dir / "revisions.json"
        self.settings = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        self.config_dir.mkdir(exist_ok=True)
        if not self.config_path.exists():
            config = json.loads(json.dumps(self.DEFAULT_CONFIG))
            config["api_keys"]["gemini"] = os.environ.get("GEMINI_API_KEY", "")
            config["api_keys"]["openai"] = os.environ.get("OPENAI_API_KEY", "")
            config["api_keys"]["xai"] = os.environ.get("XAI_API_KEY", "")
            self._write_json(self.config_path, config)
            return config

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
        except Exception as exc:
            logging.error("Failed to read config %s: %s", self.config_path, exc)
            return json.loads(json.dumps(self.DEFAULT_CONFIG))

        for key, value in self.DEFAULT_CONFIG.items():
            config.setdefault(key, value)

        for provider in ("gemini", "openai", "xai"):
            env_key = os.environ.get(f"{provider.upper()}_API_KEY")
            if env_key:
                config.setdefault("api_keys", {})[provider] = env_key

        return config

    def save(self) -> None:
        self._write_json(self.config_path, self.settings)

    def get_output_dir(self) -> Path:
        output_path = Path(self.settings.get("output_directory", "generated_documents"))
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.mkdir(exist_ok=True, parents=True)
        return output_path

    def add_to_history(self, prompt: str, filepaths: Dict[str, str], model: str, formats: Iterable[str]) -> None:
        if not self.settings.get("history_enabled", True):
            return
        try:
            history = self._read_json(self.history_path, default=[])
            entry = HistoryEntry(
                timestamp=datetime.now().isoformat(),
                prompt=prompt,
                model=model,
                filepath=filepaths.get(SupportedFormat.DOCX.value, next(iter(filepaths.values()), "")),
                formats=sorted(set(formats)),
            )
            history.append(entry.__dict__)
            max_items = int(self.settings.get("max_history_items", 100))
            history = history[-max_items:]
            self._write_json(self.history_path, history)
        except Exception as exc:
            logging.warning("Failed to update history: %s", exc)

    def add_revision(self, prompt: str, content: str, model: str) -> None:
        try:
            revisions = self._read_json(self.revision_path, default=[])
            revisions.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt,
                    "model": model,
                    "content": content,
                }
            )
            self._write_json(self.revision_path, revisions[-200:])
        except Exception as exc:
            logging.debug("Failed to persist revision history: %s", exc)

    def get_recent_history(self, limit: int = 10) -> List[HistoryEntry]:
        history_raw = self._read_json(self.history_path, default=[])
        entries = [HistoryEntry(**item) for item in history_raw[-limit:]]
        return list(reversed(entries))

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def list_models(provider: Optional[str] = None) -> Dict[str, List[str]]:
        if provider:
            return {provider: MODEL_REGISTRY.get(provider, [])}
        return MODEL_REGISTRY
