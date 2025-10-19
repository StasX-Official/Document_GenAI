"""Entry point for the AI Document Generator CLI."""
from __future__ import annotations

import sys

from cli import main as cli_main


def main(argv: list[str] | None = None) -> None:
    cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - script entry
    main(sys.argv[1:])
