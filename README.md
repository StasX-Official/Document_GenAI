# Gemini Document Generator

A command-line tool for generating professional documents using Google's Gemini AI. The tool allows you to easily create well-formatted documents from AI-generated content.

## Features

- Interactive command-line interface
- Professional document formatting
- Save history of generated documents
- Configurable settings
- Rich text UI with progress indicators (optional)
- Customizable output directory
- Multiple Gemini model support

## Installation

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key ([Get a key here](https://ai.google.dev/))

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/StasX-Official/gemini-doc-generator.git
cd gemini-doc-generator

# Install required dependencies
pip install -r requirements.txt
```

### Dependencies

The main dependencies are:

```
google-generativeai
python-docx
```

For enhanced UI experience (optional but recommended):

```
rich
```

## Configuration

The application stores its configuration in `~/.gemini_docs/config.json`. You can set your API key in one of these ways:

1. Environment variable: `export GEMINI_API_KEY=your_api_key`
2. Enter it when prompted on first run
3. Edit the config file directly

The configuration file will be automatically created on first run with default settings.

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | Your Gemini API key | Empty (will prompt) |
| `default_model` | Default Gemini model to use | "gemini-2.0-flash" |
| `output_directory` | Directory to save generated documents | "generated_documents" |
| `document_template` | Document template to use | "default" |
| `log_level` | Logging level | "INFO" |
| `history_enabled` | Enable/disable history tracking | true |
| `max_history_items` | Maximum number of items in history | 50 |
| `default_temperature` | Creativity level (0.0 to 1.0) | 0.7 |

## Usage

### Interactive Mode

Run the tool in interactive mode to enter prompts and generate documents:

```bash
python gemini_doc_generator.py
```

### Direct Prompt

Generate a document with a specific prompt and exit:

```bash
python gemini_doc_generator.py -p "Write a detailed business proposal for a renewable energy startup"
```

### Command-line Options

```
usage: gemini_doc_generator.py [-h] [-p PROMPT] [--config CONFIG] [--debug]

Gemini Document Generator

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Generate a document with this prompt and exit
  --config CONFIG       Path to custom config file
  --debug               Enable debug logging
```

## Document Output

Generated documents are saved in the configured output directory (default: `./generated_documents/`). 

Each document includes:
- A title header
- Generation metadata (timestamp, prompt)
- Well-formatted content with proper spacing and indentation
- A subtle footer

## Examples

### Basic Usage

```bash
python gemini_doc_generator.py
```

Then enter prompts like:
- "Write a research summary on renewable energy technologies"
- "Create a project plan for building a mobile application"
- "Draft a business proposal for an AI-powered healthcare service"

### Non-interactive Usage

```bash
python gemini_doc_generator.py -p "Write a comprehensive guide to machine learning algorithms"
```

## Troubleshooting

### API Key Issues

If you encounter authentication errors:
1. Verify your API key is correct
2. Check that your API key has permissions for the Gemini models
3. Ensure you're not hitting API rate limits

### Missing Rich UI

If you don't see the enhanced UI, install the optional dependency:
```bash
pip install rich
```

### Windows-Specific Issues

On Windows, if you encounter event loop errors, the application automatically uses `WindowsSelectorEventLoopPolicy()` to address common asyncio issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google Gemini AI for the content generation
- python-docx for document creation
- The Rich library for enhanced terminal UI

## Copyright
 - Copyright (c) 2025 Kozosvyst Stas (StasX)
