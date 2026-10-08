"""Live auto-transpose: hear whatever is playing, shifted onto the white keys.

Imports here are deliberately lazy. `_winaudio_helper` runs as a subprocess and
must load `comtypes` before anything loads `soundcard`, or COM apartment modes
clash; pulling the CLI in at package-import time would make that impossible.
"""
from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["main"]


def __getattr__(name: str):
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
