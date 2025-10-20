# AI Document Generator

Modernized CLI toolkit for producing documents, presentations, and structured outputs with leading AI providers.

## Highlights

- Unified provider registry with adapters for OpenAI, Google Gemini, xAI, Anthropic, Mistral, Groq, Ollama, and local Diffusers
- Configurable generation parameters (temperature, top_p, max_tokens) per request or session
- Multi-format export (DOCX, PDF, HTML, Markdown, TXT, JSON) with automatic history tracking
- Prompt batching from files or directories, interactive session commands, and secure prompt validation
- Optional image generation and image-to-text analysis routed through provider capabilities
- Video synthesis hooks for OpenAI, Runway, and Pika with configurable format and duration
- Multilingual CLI onboarding (English, Ukrainian, Russian, Polish) with persistent locale selection

## Project Layout

- `config.py` — persistent configuration manager, alias resolution, security policies
- `ai_providers.py` — provider factory and async adapters
- `security.py` — prompt and filesystem guard rails
- `outputs.py` — artifact writers for each export format
- `templates.py` — reusable renderers for DOCX, HTML, Markdown, PPTX
- `generator.py` — orchestration pipeline tying configuration, providers, and outputs together
- `cli.py` — rich CLI experience with batch and interactive workflows
- `main.py` — entry point for command-line execution

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Populate API keys via environment variables or the generated `~/.ai_docs/config.json` file. Supported keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`. Ollama and Diffusers default to local endpoints and do not require keys.

On first launch the CLI will prompt for a preferred language (English, Українська, Русский, Polski) and remember the selection for subsequent runs.

## Usage

Single-shot generation:

```powershell
python main.py --prompt "Outline the future of multimodal search" --model openai:gpt-4o --formats docx,pdf,markdown
```

Batch prompts from files:

```powershell
python main.py --prompt-dir prompts/ --model gemini:gemini-1.5-flash --temperature 0.4
```

Interactive mode with live parameter tuning:

```powershell
python main.py --model openai:gpt-4o
```

Available interactive commands: `:model`, `:formats`, `:temperature`, `:top_p`, `:max_tokens`, `:video_prompt`, `:video_format`, `:video_duration`, `:language`, `:help`.

Generate an accompanying video artifact (provider support required):

```powershell
python main.py --prompt "Storyboard a short product teaser" --model runway:gen2 --video-prompt "Cinematic teaser of a futuristic gadget" --video-format mp4 --video-duration 8
```

## Security

All prompts pass through configurable validation (length limits, deny patterns, ASCII requirements) and artifact paths are sanitized to mitigate injection or traversal attacks.

## Release Notes (v2.1.0)

- Added persistent localization with English, Ukrainian, Russian, and Polish translations plus `:language` command
- Introduced provider-agnostic video generation pipeline with configurable format and duration controls
- Extended CLI and configuration to accept `--video-*` flags and expose version information via `--version`
- Expanded output manager to persist video artifacts alongside existing document formats and JSON manifests
- Hardened generator workflow around video settings, metadata, and history tracking
- See `CHANGELOG.md` for the complete revision history

## License

MIT
