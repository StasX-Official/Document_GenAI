<img width="6912" height="3456" alt="g" src="https://github.com/user-attachments/assets/65fd6cab-ab1f-4dde-be17-6f219a65e3f8" />

# AI Document Generator

## Overview

This is a Python-based tool for generating Word documents (.docx) using various AI models from providers like Google Gemini, OpenAI (GPT), and xAI (Grok). It allows users to input prompts, generate content via AI, and save it in a templated document format. The tool supports CLI interaction, configuration via JSON, history tracking, and more.

Key features:
- Multi-provider AI support: Gemini, OpenAI, xAI (Grok).
- Asynchronous content generation for efficiency.
- Customizable templates for document formatting.
- History logging of generated documents.
- Rich console output (optional, via `rich` library).
- Secure API key handling (environment variables preferred).

## Project Structure

The code is modularized into the following files:

- **`config.py`**: Handles configuration loading, saving, and history management. Defines enums for providers and models.
- **`templates.py`**: Defines the document template for formatting generated .docx files.
- **`ai_providers.py`**: Implements AI providers for Gemini, OpenAI, and xAI. Handles content generation asynchronously.
- **`generator.py`**: Core logic for document generation, including setup, content creation, and processing prompts.
- **`cli.py`**: CLI interface for interactive mode and argument parsing.
- **`main.py`**: Entry point script to run the application.

## Requirements

- Python 3.8+ (tested on 3.12).
- Required libraries:
  - `python-docx`: For generating .docx files.
  - `google-generativeai`: For Gemini models.
  - `openai`: For GPT models.
  - `xai-sdk`: For Grok models (note: this may require specific installation; check xAI documentation).
  - `rich` (optional): For enhanced console output with colors, progress bars, etc.
  - `asyncio`: Built-in, for async operations.

Install dependencies using pip:

```bash
pip install python-docx google-generativeai openai xai-sdk rich
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

- `api_keys` - provider API keys
- `default_model` - default model identifier in `provider:model_name` format
- `output_directory` - where generated files are stored
- `default_temperature` - generation temperature

## Usage (CLI)

Start interactive mode:

```powershell
python main.py
```

Generate a single document from a prompt:

```powershell
python main.py --prompt "Write a short overview of AI" --model "openai:gpt-4o"
```

Main CLI flags:

- `--prompt, -p` prompt text
- `--model, -m` model identifier (`provider:model_name`)
- `--config` path to custom config file
- `--image-prompt` prompt to generate an image
- `--image-input` path to an image to analyze

## Output formats

- DOCX (.docx) primary output
- PPTX (.pptx) presentation output (requires `python-pptx`)
- PDF via `docx2pdf` (optional, requires system support)
- Markdown/HTML as plain text output

## Notes and limitations

- The codebase is syntactically validated. Runtime requires installing dependencies listed in `requirements.txt`.
- Provider implementations are a mix of working adapters and skeletons; actual behavior depends on provider SDKs and API keys.
- Web UI (FastAPI) is not included by default; it can be added on request.

## Quick smoke test

After installing dependencies and setting API keys, run:

```powershell
python main.py --prompt "Test document" --model "openai:gpt-4o"
```
## License

MIT
