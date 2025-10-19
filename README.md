<img width="6912" height="3456" alt="g" src="https://github.com/user-attachments/assets/65fd6cab-ab1f-4dde-be17-6f219a65e3f8" />

# AI Document Generator

## Overview

This is a Python-based tool for generating professional documents in multiple formats (DOCX, PDF, HTML, Markdown, PPTX, TXT) using AI models from providers like Google Gemini, OpenAI (GPT family), and xAI (Grok). It ships with both an upgraded CLI and a FastAPI web dashboard, includes a revision workflow, and keeps a searchable history of generated artefacts.

Key features:
- Multi-provider AI support: Gemini, OpenAI (GPT-4o, GPT-4.1, o1 and more), xAI Grok, plus a local Stable Diffusion image adapter.
- FastAPI web interface with an in-browser editor, keyboard shortcuts, and live history.
- Launcher utility to choose between the CLI and the web UI.
- Asynchronous content generation with optional image creation and vision analysis when supported by the provider.
- Interactive revision workflow that lets you refine generated copy before export and stores revision snapshots.
- Customisable templates for Word, HTML, Markdown, and presentation exports.
- History logging of generated documents and format types.
- Rich console output (optional, via `rich` library) and configurable log levels.

## Project Structure

The code is modularized into the following files:

- **`config.py`**: Handles configuration loading, saving, revision/history tracking and exposes provider/model catalogues.
- **`templates.py`**: Provides templating helpers for DOCX, HTML, Markdown and PPTX exports.
- **`ai_providers.py`**: Implements AI providers for Gemini, OpenAI, xAI (Grok) and local Diffusers, including capability metadata.
- **`generator.py`**: Core logic for content generation, revision workflow, and multi-format exporting.
- **`cli.py`**: Enhanced CLI interface with history/model listing, multi-format output, and optional editing step.
- **`web_app.py`**: FastAPI application serving the interactive dashboard.
- **`launcher.py`**: Convenience launcher that prompts for CLI or web mode.
- **`main.py`**: Thin entry point delegating to the CLI.

## Requirements

- Python 3.8+ (tested on 3.12).
- Required libraries:
- `python-docx`, `python-pptx`, `docx2pdf`: Office exports.
- `google-generativeai`, `openai`, `xai-sdk`: Gemini, GPT, Grok providers.
- `fastapi`, `uvicorn`, `jinja2`: Web dashboard.
- `rich`: Enhanced console output (optional).
- `diffusers`, `torch`: Local Stable Diffusion image generation (optional but supported).

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

**Note**: Some libraries like `xai-sdk` might not be publicly available or could require authentication. Replace or mock as needed if not accessible.

## Configuration

Configuration is stored in `~/.ai_docs/config.json` (created automatically on first run). Default settings:

# Document_GenAI

A concise, production-oriented tool for generating documents (.docx, .pptx) using multiple AI providers (OpenAI, Google Gemini, xAI, local diffusers).

## Summary

- Command-line interface for generating documents from prompts.
- Multi-provider architecture with provider adapters implemented as skeletons.
- Template-driven DOCX output, table export, and PPTX generation.
- Optional image generation and image-based analysis when providers support it.

## Installation

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure API keys via environment variables or the generated config file.

## Configuration

The application creates a config file at `~/.ai_docs/config.json` on first run. Example environment variables (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your_gemini_key"
$env:OPENAI_API_KEY = "your_openai_key"
$env:XAI_API_KEY = "your_xai_key"
```

Important config keys:

- `api_keys` – provider API keys
- `default_model` – default model identifier in `provider:model_name` format
- `output_directory` – where generated files are stored
- `default_temperature` – generation temperature
- `default_formats` – default export formats (e.g. `docx`, `pdf`, ...)
- `auto_open_editor` – whether the CLI opens the revision workflow automatically

## Usage (CLI)

Launch the interactive CLI:

```powershell
python main.py
```

Generate a document bundle in several formats:

```powershell
python main.py --prompt "Write a short overview of AI" --model "openai:gpt-4o" --format docx,pdf,html
```

Main CLI flags:

- `--prompt, -p` prompt text
- `--model, -m` model identifier (`provider:model_name`)
- `--config` path to custom config file
- `--image-prompt` prompt to generate an image
- `--image-input` path to an image to analyze
- `--format` comma-separated list of output formats (`docx,pdf,html,md,txt,pptx`)
- `--list-models` optionally filtered provider for available models
- `--history` show the most recent generated artefacts
- `--no-edit` skip the CLI revision editor

Run the web dashboard:

```bash
python -m web_app  # or: uvicorn web_app:app --reload
```

Use the launcher to choose interactively:

```bash
python launcher.py
```

## Output formats

- DOCX (.docx)
- PDF (via `docx2pdf` when available)
- HTML (responsive single-page document)
- Markdown (.md)
- Plain text (.txt)
- PowerPoint (.pptx) summarising the generated sections

## Notes and limitations

- The codebase is syntactically validated. Runtime requires installing dependencies listed in `requirements.txt`.
- Provider implementations are a mix of working adapters and skeletons; actual behavior depends on provider SDKs and API keys.
- The FastAPI web UI serves local files (`file://` links). On remote hosts consider mounting a shared volume or enabling downloads via another channel.

## Quick smoke test

After installing dependencies and setting API keys, run:

```powershell
python main.py --prompt "Test document" --model "openai:gpt-4o"
```
## License

MIT
