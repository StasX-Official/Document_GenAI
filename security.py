from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import re
import secrets


@dataclass
class SecurityResult:
    valid: bool
    reason: Optional[str] = None


class SecurityValidator:
    def __init__(
        self,
        deny_patterns: Sequence[str],
        max_prompt_length: int,
        require_ascii: bool,
        forbidden_paths: Sequence[str],
        blocked_extensions: Sequence[str],
    ):
        self._deny_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in deny_patterns]
        self._max_prompt_length = max_prompt_length
        self._require_ascii = require_ascii
        self._forbidden_paths = [Path(path).expanduser().resolve() for path in forbidden_paths]
        self._blocked_extensions = {ext.lower() for ext in blocked_extensions}

    def validate_prompt(self, prompt: str) -> SecurityResult:
        if not prompt or not prompt.strip():
            return SecurityResult(False, "Prompt is empty")
        if self._require_ascii and not prompt.isascii():
            return SecurityResult(False, "Prompt must be ASCII")
        if len(prompt) > self._max_prompt_length:
            return SecurityResult(False, "Prompt exceeds maximum length")
        for pattern in self._deny_patterns:
            if pattern.search(prompt):
                return SecurityResult(False, "Prompt contains blocked instructions")
        if self._looks_like_secret(prompt):
            return SecurityResult(False, "Prompt appears to include secrets")
        return SecurityResult(True)

    def validate_path(self, path: Path) -> SecurityResult:
        resolved = path.expanduser().resolve()
        for forbidden in self._forbidden_paths:
            try:
                resolved.relative_to(forbidden)
                return SecurityResult(False, "Path targets a restricted location")
            except ValueError:
                continue
        if resolved.suffix.lower() in self._blocked_extensions:
            return SecurityResult(False, "File extension is blocked")
        return SecurityResult(True)

    def sanitize_filename(self, value: str) -> str:
        token = secrets.token_hex(4)
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
        if not cleaned:
            cleaned = "document"
        return f"{cleaned[:48]}-{token}"

    def _looks_like_secret(self, text: str) -> bool:
        entropy_segments = re.findall(r"[A-Za-z0-9]{24,}", text)
        if entropy_segments:
            return True
        keywords = ["api_key", "secret", "token", "password", "ssh-rsa", "BEGIN"]
        return any(keyword.lower() in text.lower() for keyword in keywords)

    def allowed_formats(self, formats: Iterable[str]) -> List[str]:
        whitelist = {"docx", "pdf", "html", "markdown", "md", "txt", "json"}
        return [fmt for fmt in formats if fmt.lower() in whitelist]
