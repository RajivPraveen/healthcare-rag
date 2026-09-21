"""Rich-backed logging used by the CLI and services."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

console = Console()
_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=(level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # These are chatty on first model download.
    for noisy in ("httpx", "sentence_transformers", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
