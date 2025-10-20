import argparse
from pathlib import Path
from typing import List, Optional

from generator import DocumentGenerator
from loc.language import LanguageManager, resolve_language_code

try:
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


def _parse_formats(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    return items or None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


async def run_cli(
    generator: DocumentGenerator,
    initial_model: Optional[str],
    localization: LanguageManager,
    initial_video_prompt: Optional[str] = None,
    initial_video_format: Optional[str] = None,
    initial_video_duration: Optional[int] = None,
) -> None:
    state = {
        "model": initial_model,
        "formats": None,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "video_prompt": initial_video_prompt,
        "video_format": initial_video_format.strip().lower().lstrip(".") if initial_video_format else None,
        "video_duration": initial_video_duration,
    }

    def _render_header() -> None:
        banner = localization.translate("app.banner")
        mode_label = localization.translate("app.mode")
        help_hint = localization.translate("app.help_hint")
        if RICH_AVAILABLE and generator.console:
            generator.console.print(f"[bold]{banner}[/bold] [dim]{mode_label}[/dim]")
            generator.console.print(help_hint)
        else:
            print(f"{banner} ({mode_label})")
            print(help_hint)

    _render_header()
    while True:
        prompt_label = localization.translate("prompt.label")
        try:
            if RICH_AVAILABLE and generator.console:
                user_input = Prompt.ask(prompt_label)
            else:
                user_input = input(f"{prompt_label}: ")
        except KeyboardInterrupt:
            print(f"\n{localization.translate('prompt.stop')}")
            break
        command = user_input.strip()
        if not command:
            continue
        if command.lower() in {"exit", "quit", "q"}:
            break
        if command.startswith(":"):
            rerender = _handle_interactive_command(command, state, generator, localization)
            if rerender:
                _render_header()
            continue
        await generator.process_prompt(
            command,
            state["model"],
            formats=state["formats"],
            temperature=state["temperature"],
            top_p=state["top_p"],
            max_tokens=state["max_tokens"],
            video_prompt=state["video_prompt"],
            video_format=state["video_format"],
            video_duration=state["video_duration"],
        )


def _handle_interactive_command(command: str, state: dict, generator: DocumentGenerator, localization: LanguageManager) -> bool:
    parts = command[1:].split()
    if not parts:
        return False
    key = parts[0].lower()
    value = " ".join(parts[1:]) if len(parts) > 1 else None
    if key == "model" and value:
        state["model"] = value
        generator.logger.info(localization.translate("info.model_set", model=value))
    elif key == "formats" and value:
        state["formats"] = [item.strip() for item in value.split(",") if item.strip()]
        formats_str = ", ".join(state["formats"] or [])
        generator.logger.info(localization.translate("info.formats_set", formats=formats_str))
    elif key == "temperature" and value:
        state["temperature"] = _parse_float(value)
        generator.logger.info(localization.translate("info.temperature_set", temperature=state["temperature"]))
    elif key == "top_p" and value:
        state["top_p"] = _parse_float(value)
        generator.logger.info(localization.translate("info.top_p_set", top_p=state["top_p"]))
    elif key == "max_tokens" and value:
        state["max_tokens"] = _parse_int(value)
        generator.logger.info(localization.translate("info.max_tokens_set", max_tokens=state["max_tokens"]))
    elif key == "help":
        message = localization.translate("help.commands")
        if RICH_AVAILABLE and generator.console:
            generator.console.print(message)
        else:
            print(message)
    elif key == "video_prompt" and value:
        state["video_prompt"] = value
        generator.logger.info(localization.translate("info.video_prompt_set", video_prompt=value))
    elif key == "video_format" and value:
        state["video_format"] = value.strip().lower().lstrip(".")
        generator.logger.info(localization.translate("info.video_format_set", video_format=state["video_format"]))
    elif key == "video_duration" and value:
        state["video_duration"] = _parse_int(value)
        generator.logger.info(localization.translate("info.video_duration_set", video_duration=state["video_duration"]))
    elif key == "language" and value:
        code = resolve_language_code(value)
        if not code:
            generator.logger.info(localization.translate("info.invalid_language"))
            return False
        localization.set_language(code)
        generator.config.set_language(code)
        generator.config.save()
        generator.logger.info(localization.translate("info.language_set", language=localization.language_name))
        return True
    else:
        generator.logger.info(localization.translate("info.unknown_command"))
    return False


async def async_main(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else None
    generator = DocumentGenerator(config_path)
    localization = _initialize_localization(generator)
    if args.list_models:
        print(localization.translate("info.available_models"))
        for name in generator.config.list_known_models():
            print(name)
        return
    if args.set_default_model:
        generator.config.override_defaults(model=args.set_default_model)
        generator.config.save()
        print(localization.translate("info.default_model_updated", model=args.set_default_model))
        return
    formats = _parse_formats(args.formats)
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    video_prompt = args.video_prompt
    video_format = args.video_format
    video_duration = args.video_duration
    if args.prompt or args.prompt_file or args.prompt_dir:
        await generator.process_prompt(
            args.prompt or "",
            args.model,
            prompt_file=args.prompt_file,
            prompt_dir=args.prompt_dir,
            formats=formats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            image_input=args.image_input,
            image_prompt=args.image_prompt,
            video_prompt=video_prompt,
            video_format=video_format,
            video_duration=video_duration,
        )
        return
    await run_cli(
        generator,
        args.model,
        localization,
        initial_video_prompt=video_prompt,
        initial_video_format=video_format,
        initial_video_duration=video_duration,
    )


def _initialize_localization(generator: DocumentGenerator) -> LanguageManager:
    config_lang = generator.config.language
    localization = LanguageManager(config_lang)
    canonical = resolve_language_code(config_lang)
    needs_prompt = generator.config.is_first_run or not config_lang or not canonical
    if not needs_prompt:
        localization.set_language(canonical)
        return localization
    selected_code = _prompt_language_selection(localization, generator.console if RICH_AVAILABLE else None)
    localization.set_language(selected_code)
    generator.config.set_language(localization.code)
    generator.config.save()
    generator.logger.info(localization.translate("info.language_set", language=localization.language_name))
    return localization


def _prompt_language_selection(localization: LanguageManager, console) -> str:
    prompt_text = "Select language / Оберіть мову / Выберите язык / Wybierz język"
    error_text = "Invalid choice / Невірний вибір / Неверный выбор / Nieprawidłowy wybór"
    options = list(localization.available_languages().items())

    def _emit(message: str) -> None:
        if console is not None and RICH_AVAILABLE:
            console.print(message)
        else:
            print(message)

    _emit(prompt_text)
    for index, (code, label) in enumerate(options, start=1):
        _emit(f"{index}. {label} ({code})")

    while True:
        if console is not None and RICH_AVAILABLE:
            choice = Prompt.ask(prompt_text).strip()
        else:
            choice = input(f"{prompt_text}: ").strip()
        normalized = choice.lower()
        if not normalized:
            return options[0][0]
        if normalized.isdigit():
            idx = int(normalized) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        canonical = resolve_language_code(normalized)
        if canonical:
            return canonical
        _emit(error_text)