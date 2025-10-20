import argparse
import asyncio
import logging
import os
import sys
import time

from cli import async_main
from version import __version__

def main():
    parser = argparse.ArgumentParser(description=f"AI Document Generator v{__version__}")
    parser.add_argument("-p", "--prompt", help="Generate a document with this prompt and exit")
    parser.add_argument("-m", "--model", help="Model to use in provider:model format or alias")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--prompt-file", help="Path to a file containing a prompt or list of prompts")
    parser.add_argument("--prompt-dir", help="Directory with prompt files (.txt, .md, .json)")
    parser.add_argument("--image-prompt", help="Generate image from this prompt and include in the document")
    parser.add_argument("--image-input", help="Path to an input image to analyze")
    parser.add_argument("--video-prompt", help="Generate a video from this prompt and save it")
    parser.add_argument("--video-format", help="Video output format (default from config)")
    parser.add_argument("--video-duration", type=int, help="Video duration in seconds")
    parser.add_argument("--formats", help="Comma separated output formats, e.g. docx,pdf,html")
    parser.add_argument("--temperature", type=float, help="Override temperature for generation")
    parser.add_argument("--top-p", type=float, help="Override nucleus sampling top_p")
    parser.add_argument("--max-tokens", type=int, help="Override response token limit")
    parser.add_argument("--list-models", action="store_true", help="List known model aliases from config")
    parser.add_argument("--set-default-model", help="Update default model alias in config and exit")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ["LOG_LEVEL"] = "DEBUG"

    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        asyncio.run(async_main(args))
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
    finally:
        print("Shutting down...")
        time.sleep(0.5)

if __name__ == "__main__":
    main()