import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

class ProviderType(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    XAI = "xai"
    DIFFUSERS = "diffusers"

class ModelType(str, Enum):
    GEMINI_PRO = "gemini:gemini-1.5-pro"
    GEMINI_FLASH = "gemini:gemini-1.5-flash"
    GPT_4O = "openai:gpt-4o"
    GPT_4O_MINI = "openai:gpt-4o-mini"
    GROK_3 = "xai:grok-3"
    GROK_3_MINI = "xai:grok-3-mini"
    DIFFUSERS_LOCAL = "diffusers:local"

class Config:
    DEFAULT_CONFIG = {
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
        "max_history_items": 50,
        "default_temperature": 0.7,
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_dir = Path.home() / ".ai_docs"
        self.config_path = config_path or self.config_dir / "config.json"
        self.history_path = self.config_dir / "history.json"
        self.settings = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        self.config_dir.mkdir(exist_ok=True)

        if not self.config_path.exists():
            config = self.DEFAULT_CONFIG.copy()
            config["api_keys"]["gemini"] = os.environ.get("GEMINI_API_KEY", "")
            config["api_keys"]["openai"] = os.environ.get("OPENAI_API_KEY", "")
            config["api_keys"]["xai"] = os.environ.get("XAI_API_KEY", "")

            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)

            return config

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            for key, value in self.DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value

            env_keys = {
                "gemini": os.environ.get("GEMINI_API_KEY"),
                "openai": os.environ.get("OPENAI_API_KEY"),
                "xai": os.environ.get("XAI_API_KEY"),
            }
            for provider, env_key in env_keys.items():
                if env_key:
                    config["api_keys"][provider] = env_key

            return config
        except Exception as e:
            logging.error(f"Failed to load config: {e}. Using defaults.")
            return self.DEFAULT_CONFIG.copy()

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.settings, f, indent=2)

    def get_output_dir(self) -> Path:
        output_path = Path(self.settings["output_directory"])
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.mkdir(exist_ok=True, parents=True)
        return output_path

    def add_to_history(self, prompt: str, filepath: str, model: str) -> None:
        if not self.settings.get("history_enabled", True):
            return

        try:
            history: List[Dict[str, Any]] = []
            if self.history_path.exists():
                with open(self.history_path, "r") as f:
                    history = json.load(f)

            history.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "model": model,
                "filepath": filepath,
            })

            max_items = self.settings.get("max_history_items", 50)
            history = history[-max_items:]

            with open(self.history_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to update history: {e}")