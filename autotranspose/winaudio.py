"""Mute devices and switch the Windows default output, so you do not have to.

Everything here shells out to `_winaudio_helper` in a separate process; see that
module for why. Each call costs a few hundred milliseconds, which is fine for
something that happens twice per session.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass


class WinAudioUnavailable(RuntimeError):
    """Core Audio control is not usable (not Windows, or pycaw missing)."""


@dataclass
class Endpoint:
    id: str
    name: str
    muted: bool
    volume: float


def _call(*args: str) -> dict:
    cmd = [sys.executable, "-m", "autotranspose._winaudio_helper", *args]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=25,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WinAudioUnavailable(str(exc)) from None
    out = (proc.stdout or "").strip().splitlines()
    for line in reversed(out):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if not data.get("ok"):
            raise WinAudioUnavailable(data.get("error", "unknown error"))
        return data
    raise WinAudioUnavailable(
        (proc.stderr or "no output from the audio helper").strip().splitlines()[-1]
        if proc.stderr else "no output from the audio helper"
    )


def available() -> bool:
    try:
        _call("list")
        return True
    except WinAudioUnavailable:
        return False


def endpoints() -> list[Endpoint]:
    """Every active endpoint with a volume control, render and capture alike.

    Which of them are outputs is a question `devices.list_outputs()` already
    answers, so this does not try to classify them.
    """
    return [
        Endpoint(id=d["id"], name=d["name"], muted=d["muted"], volume=d["volume"])
        for d in _call("list")["devices"]
    ]


def find(endpoint_id: str) -> Endpoint | None:
    return next((e for e in endpoints() if e.id == endpoint_id), None)


def set_default_output(endpoint_id: str) -> None:
    _call("set-default", endpoint_id)


def set_mute(endpoint_id: str, muted: bool) -> None:
    _call("set-mute", endpoint_id, "1" if muted else "0")
