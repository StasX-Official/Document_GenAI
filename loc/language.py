from importlib import import_module
from typing import Dict, Iterable, Tuple

LANGUAGE_ORDER: Tuple[str, ...] = ("en", "uk", "ru", "pl")
LANGUAGE_MAP: Dict[str, Tuple[str, str]] = {
    "en": ("english", "English"),
    "uk": ("ukrainian", "Українська"),
    "ua": ("ukrainian", "Українська"),
    "ru": ("russian", "Русский"),
    "pl": ("polish", "Polski"),
    "eng": ("english", "English"),
    "eng-us": ("english", "English"),
    "eng-gb": ("english", "English"),
    "rus": ("russian", "Русский"),
    "pol": ("polish", "Polski"),
}
MODULE_PRIMARY: Dict[str, str] = {
    "english": "en",
    "ukrainian": "uk",
    "russian": "ru",
    "polish": "pl",
}


def _resolve_code(code: str) -> str:
    if not code:
        return "en"
    normalized = code.strip().lower()
    if normalized in LANGUAGE_MAP:
        module_name = LANGUAGE_MAP[normalized][0]
        return MODULE_PRIMARY.get(module_name, "en")
    return ""


def resolve_language_code(code: str) -> str:
    return _resolve_code(code)


class LanguageManager:
    """Simple localization lookup helper."""

    def __init__(self, code: str = "en") -> None:
        self._code = "en"
        self._language_name = "English"
        self._strings: Dict[str, str] = {}
        self.set_language(code)

    def set_language(self, code: str) -> str:
        canonical = _resolve_code(code)
        if not canonical:
            canonical = "en"
        module_name, language_name = LANGUAGE_MAP.get(canonical, LANGUAGE_MAP["en"])
        try:
            module = import_module(f"loc.{module_name}")
            strings = getattr(module, "STRINGS", {})
            if not isinstance(strings, dict):
                raise TypeError("Localization STRINGS must be a dict")
        except Exception:
            module = import_module("loc.english")
            strings = getattr(module, "STRINGS", {})
            canonical = "en"
            module_name, language_name = LANGUAGE_MAP[canonical]
        self._code = canonical
        self._language_name = language_name
        self._strings = dict(strings)
        return self._code

    def translate(self, key: str, **params: str) -> str:
        template = self._strings.get(key, key)
        if params:
            try:
                return template.format(**params)
            except Exception:
                return template
        return template

    def available_languages(self) -> Dict[str, str]:
        available: Dict[str, str] = {}
        for code in LANGUAGE_ORDER:
            module = LANGUAGE_MAP.get(code)
            if module:
                available[code] = module[1]
        return available

    def supported_codes(self) -> Iterable[str]:
        return tuple(self.available_languages().keys())

    @property
    def code(self) -> str:
        return self._code

    @property
    def language_name(self) -> str:
        return self._language_name

    @property
    def strings(self) -> Dict[str, str]:
        return dict(self._strings)
