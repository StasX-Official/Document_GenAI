import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class ProviderType(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    XAI = "xai"
    DIFFUSERS = "diffusers"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    GROQ = "groq"
    OLLAMA = "ollama"
    STABILITY = "stability"
    LEONARDO = "leonardo"
    RUNWAY = "runway"
    PIKA = "pika"
    REPLICATE = "replicate"


@dataclass
class SecurityConfig:
    deny_patterns: List[str] = field(default_factory=list)
    max_prompt_length: int = 8000
    require_ascii: bool = False
    forbidden_paths: List[str] = field(default_factory=list)
    blocked_extensions: List[str] = field(default_factory=lambda: [".exe", ".bat", ".sh", ".ps1", ".cmd"])


@dataclass
class ConfigSnapshot:
    api_keys: Dict[str, str]
    default_model: str
    output_directory: str
    default_formats: List[str]
    default_video_format: str
    default_video_duration: int
    log_level: str
    history_enabled: bool
    max_history_items: int
    default_temperature: float
    default_top_p: Optional[float]
    default_max_tokens: Optional[int]
    model_aliases: Dict[str, str]
    security: SecurityConfig
    language: str


class ConfigManager:
    DEFAULTS: Dict[str, Any] = {
        "api_keys": {
            "gemini": "",
            "openai": "",
            "xai": "",
            "anthropic": "",
            "mistral": "",
            "groq": "",
            "ollama": "",
            "stability": "",
            "leonardo": "",
            "runway": "",
            "pika": "",
            "replicate": "",
        },
        "default_model": "openai:gpt-4o-mini",
        "output_directory": "generated_documents",
        "default_formats": ["docx", "pdf", "markdown", "html"],
        "default_video_format": "mp4",
    "default_video_duration": 8,
        "log_level": "INFO",
        "history_enabled": True,
        "max_history_items": 100,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "default_max_tokens": 2048,
        "language": "en",
        "model_aliases": {
            "gpt4": "openai:gpt-4o",
            "gpt4o": "openai:gpt-4o",
            "gpt4o-mini": "openai:gpt-4o-mini",
            "gemini-flash": "gemini:gemini-1.5-flash",
            "gemini-pro": "gemini:gemini-1.5-pro",
            "grok": "xai:grok-3",
            "grok-mini": "xai:grok-3-mini",
            "claude": "anthropic:claude-3-sonnet-20240229",
            "mistral-large": "mistral:mistral-large-latest",
            "groq-mixtral": "groq:mixtral-8x7b-32768",
            "ollama-llama3": "ollama:llama3",
            "sdxl": "stability:sdxl-turbo",
            "leonardo-creative": "leonardo:creative-v1",
            "runway-gen2": "runway:gen2",
            "pika-1": "pika:pika-1",
            "replicate-llama3": "replicate:meta/llama-3-70b",
        },
        "security": {
            "deny_patterns": [
                r"rm\s+-rf",
                r"shutdown\s+-",
                r"format\s+c:",
                r"powershell\s+-nop",
                r"curl\s+.*\|\s+sh",
            ],
            "max_prompt_length": 12000,
            "require_ascii": False,
            "forbidden_paths": [],
            "blocked_extensions": [".exe", ".bat", ".sh", ".ps1", ".cmd"],
        },
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_dir = Path.home() / ".ai_docs"
        self.config_path = config_path or self.config_dir / "config.json"
        self.history_path = self.config_dir / "history.json"
        self._first_run = False
        self._snapshot = self._load()

    @property
    def snapshot(self) -> ConfigSnapshot:
        return self._snapshot

    def _load(self) -> ConfigSnapshot:
        self.config_dir.mkdir(exist_ok=True)
        data = self._read_config_file()
        data = self._merge_with_defaults(data)
        security = self._build_security_config(data.get("security", {}))
        snapshot = ConfigSnapshot(
            api_keys=data["api_keys"],
            default_model=data["default_model"],
            output_directory=data["output_directory"],
            default_formats=list(dict.fromkeys(data.get("default_formats", []))),
            default_video_format=self._normalize_video_format(data.get("default_video_format", "mp4")),
            default_video_duration=int(data.get("default_video_duration", 8)),
            log_level=data["log_level"],
            history_enabled=data["history_enabled"],
            max_history_items=data["max_history_items"],
            default_temperature=float(data["default_temperature"]),
            default_top_p=self._try_float(data.get("default_top_p")),
            default_max_tokens=self._try_int(data.get("default_max_tokens")),
            model_aliases=data.get("model_aliases", {}),
            security=security,
            language=data.get("language", "en"),
        )
        return snapshot

    def _read_config_file(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            self._first_run = True
            data = json.loads(json.dumps(self.DEFAULTS))
            self._inject_env_keys(data["api_keys"])
            self._write_file(data)
            return data
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._inject_env_keys(data.setdefault("api_keys", {}))
            self._first_run = False
            return data
        except Exception as exc:
            logging.error(f"Config load failed: {exc}")
            data = json.loads(json.dumps(self.DEFAULTS))
            self._inject_env_keys(data["api_keys"])
            self._first_run = True
            return data

    def _inject_env_keys(self, api_keys: Dict[str, str]) -> None:
        for key in api_keys:
            env_key = os.environ.get(key.upper() + "_API_KEY")
            if env_key:
                api_keys[key] = env_key

    def _merge_with_defaults(self, data: Dict[str, Any]) -> Dict[str, Any]:
        merged = json.loads(json.dumps(self.DEFAULTS))
        for key, value in data.items():
            merged[key] = value
        merged["api_keys"].update(data.get("api_keys", {}))
        merged["model_aliases"].update(data.get("model_aliases", {}))
        merged["security"].update(data.get("security", {}))
        merged["default_video_format"] = self._normalize_video_format(merged.get("default_video_format"))
        return merged

    def _write_file(self, data: Dict[str, Any]) -> None:
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _build_security_config(self, data: Dict[str, Any]) -> SecurityConfig:
        return SecurityConfig(
            deny_patterns=list(data.get("deny_patterns", [])),
            max_prompt_length=int(data.get("max_prompt_length", 12000)),
            require_ascii=bool(data.get("require_ascii", False)),
            forbidden_paths=list(data.get("forbidden_paths", [])),
            blocked_extensions=list(data.get("blocked_extensions", [".exe", ".bat", ".sh", ".ps1", ".cmd"])),
        )

    def refresh(self) -> None:
        self._snapshot = self._load()

    def save(self) -> None:
        data = {
            "api_keys": self._snapshot.api_keys,
            "default_model": self._snapshot.default_model,
            "output_directory": self._snapshot.output_directory,
            "default_formats": self._snapshot.default_formats,
            "default_video_format": self._snapshot.default_video_format,
            "default_video_duration": self._snapshot.default_video_duration,
            "log_level": self._snapshot.log_level,
            "history_enabled": self._snapshot.history_enabled,
            "max_history_items": self._snapshot.max_history_items,
            "default_temperature": self._snapshot.default_temperature,
            "default_top_p": self._snapshot.default_top_p,
            "default_max_tokens": self._snapshot.default_max_tokens,
            "model_aliases": self._snapshot.model_aliases,
            "security": {
                "deny_patterns": self._snapshot.security.deny_patterns,
                "max_prompt_length": self._snapshot.security.max_prompt_length,
                "require_ascii": self._snapshot.security.require_ascii,
                "forbidden_paths": self._snapshot.security.forbidden_paths,
                "blocked_extensions": self._snapshot.security.blocked_extensions,
            },
            "language": self._snapshot.language,
        }
        self._write_file(data)

    def get_output_dir(self) -> Path:
        base = Path(self._snapshot.output_directory)
        if not base.is_absolute():
            base = Path.cwd() / base
        base.mkdir(parents=True, exist_ok=True)
        return base

    def record_history(self, prompt: str, result_paths: Dict[str, str], model: str) -> None:
        if not self._snapshot.history_enabled:
            return
        try:
            history: List[Dict[str, Any]] = []
            if self.history_path.exists():
                with open(self.history_path, "r", encoding="utf-8") as handle:
                    history = json.load(handle)
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "prompt": prompt,
                "model": model,
                "artifacts": result_paths,
            }
            history.append(entry)
            history = history[-self._snapshot.max_history_items :]
            with open(self.history_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
        except Exception as exc:
            logging.warning(f"History write failed: {exc}")

    def resolve_model(self, model: Optional[str]) -> Tuple[str, str]:
        candidate = model or self._snapshot.default_model
        candidate = self._snapshot.model_aliases.get(candidate, candidate)
        if ":" not in candidate:
            raise ValueError("Model string must include provider prefix, for example openai:gpt-4o")
        provider, provider_model = candidate.split(":", 1)
        provider = provider.strip().lower()
        if provider not in self._snapshot.api_keys:
            raise ValueError(f"Unsupported provider: {provider}")
        return provider, provider_model

    def get_api_key(self, provider: str) -> str:
        return self._snapshot.api_keys.get(provider, "")

    def resolve_formats(self, requested: Optional[Sequence[str]]) -> List[str]:
        formats = list(dict.fromkeys(requested or self._snapshot.default_formats))
        if not formats:
            formats = list(self._snapshot.default_formats)
        return [f.lower() for f in formats]

    def resolve_video_format(self, requested: Optional[str]) -> str:
        if requested and requested.strip():
            return self._normalize_video_format(requested)
        return self._snapshot.default_video_format

    def override_defaults(
        self,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        video_format: Optional[str] = None,
        video_duration: Optional[int] = None,
    ) -> None:
        if model:
            self._snapshot.default_model = model
        if temperature is not None:
            self._snapshot.default_temperature = float(temperature)
        if top_p is not None:
            self._snapshot.default_top_p = float(top_p)
        if max_tokens is not None:
            self._snapshot.default_max_tokens = int(max_tokens)
        if video_format:
            self._snapshot.default_video_format = self._normalize_video_format(video_format)
        if video_duration is not None:
            self._snapshot.default_video_duration = int(video_duration)

    def get_generation_defaults(self) -> Tuple[float, Optional[float], Optional[int]]:
        return (
            self._snapshot.default_temperature,
            self._snapshot.default_top_p,
            self._snapshot.default_max_tokens,
        )

    def get_default_video_format(self) -> str:
        return self._snapshot.default_video_format

    def get_default_video_duration(self) -> int:
        return self._snapshot.default_video_duration

    def get_security_config(self) -> SecurityConfig:
        return self._snapshot.security

    def list_known_models(self) -> Iterable[str]:
        return sorted({self._snapshot.default_model, *self._snapshot.model_aliases.values()})

    @property
    def language(self) -> str:
        return self._snapshot.language

    @property
    def is_first_run(self) -> bool:
        return self._first_run

    def set_language(self, code: str) -> None:
        self._snapshot.language = code

    def _normalize_video_format(self, value: Optional[str]) -> str:
        text = (value or "mp4").strip().lower()
        text = text.lstrip(".")
        return text or "mp4"

    def _try_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _try_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None