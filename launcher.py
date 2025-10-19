"""Launcher script that lets the user choose between CLI and web interfaces."""
from __future__ import annotations

import sys
import threading

from cli import main as cli_main
from web_app import start as start_web


def _run_web() -> None:
    print("Запускаємо веб-інтерфейс на http://127.0.0.1:8000 ...")
    start_web()


def main(argv: list[str] | None = None) -> None:
    print("=== AI Document Generator Launcher ===")
    print("1) Веб-інтерфейс")
    print("2) CLI")
    print("q) Вихід")
    choice = input("Оберіть режим (1/2/q): ").strip().lower()

    if choice == "1":
        thread = threading.Thread(target=_run_web, daemon=False)
        thread.start()
        try:
            thread.join()
        except KeyboardInterrupt:
            print("\nЗупинка веб-сервера...")
    elif choice == "2":
        cli_main(argv)
    else:
        print("Вихід.")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
