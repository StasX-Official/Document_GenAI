import argparse
import asyncio
import logging
import os
import sys
import time

from cli import async_main
from config import ModelType

def main():
    parser = argparse.ArgumentParser(description="AI Document Generator")
    parser.add_argument("-p", "--prompt", help="Generate a document with this prompt and exit")
    parser.add_argument("-m", "--model", choices=[m.value for m in ModelType], help="Model to use")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--format", choices=["docx","pdf","html","md"], default="docx", help="Output format")
    parser.add_argument("--prompt-file", help="Path to a file containing a prompt or list of prompts")
    parser.add_argument("--image-prompt", help="Generate image from this prompt and include in the document")
    parser.add_argument("--image-input", help="Path to an input image to analyze")
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