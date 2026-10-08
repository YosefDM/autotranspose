"""Arrange the audio routing automatically, and put it back afterwards.

Doing this by hand means visiting Windows sound settings, picking which device
apps play to, and muting it. All three are things Core Audio will do on request,
so the program does them:

  1. pick the device you want to *hear* from (your current default -- it is
     default because that is what you listen on),
  2. make the *other* output the system default, so apps play into it,
  3. mute that device, because muting does not affect what loopback capture
     hears (verified: 0.967x of full level muted, versus 0.009x at 5% volume --
     so mute, never turn down),
  4. capture its loopback, transpose, play to the device from step 1.

Everything changed is recorded and restored on exit, including on Ctrl-C.
"""
from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import devices, winaudio


def _state_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return Path(base) / "autotranspose" / "routing_state.json"


def _pid_alive(pid: int) -> bool:
    """Is that process still running? Used to tell a crash from a live instance."""
    if not pid or pid == os.getpid():
        return pid == os.getpid()
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    WAIT_TIMEOUT = 0x102
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        # Still running means "has not signalled", i.e. the wait times out.
        return k32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        k32.CloseHandle(handle)


def owned_by_live_process() -> int | None:
    """PID of another running instance that owns the undo record, if any."""
    data = pending_restore()
    if not data:
        return None
    pid = int(data.get("pid") or 0)
    if pid and pid != os.getpid() and _pid_alive(pid):
        return pid
    return None


def save_pending_restore(data: dict) -> None:
    """Record what we changed, so a hard kill is still recoverable.

    `atexit` does not run if the process is killed outright, and this program
    changes real system settings, so the undo information goes on disk too.
    """
    try:
        p = _state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # best effort; the in-process restore is the primary path


def clear_pending_restore() -> None:
    try:
        _state_path().unlink(missing_ok=True)
    except OSError:
        pass


def pending_restore() -> dict | None:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def restore_pending(force: bool = False) -> list[str]:
    """Undo a previous run that did not get to clean up. Returns what it did.

    Refuses if the record belongs to an instance that is still running, which
    would otherwise pull the routing out from under it.
    """
    data = pending_restore()
    if not data:
        return []
    if not force:
        other = owned_by_live_process()
        if other:
            return [
                f"skipped: another autotranspose (pid {other}) is running and owns "
                f"these settings; quit it first, or pass --force"
            ]
    done = []
    try:
        if data.get("unmute_id"):
            winaudio.set_mute(data["unmute_id"], False)
            done.append(f"unmuted {data.get('unmute_name', data['unmute_id'])}")
        if data.get("default_id"):
            winaudio.set_default_output(data["default_id"])
            done.append(f"restored default output to {data.get('default_name', '')}".strip())
    except winaudio.WinAudioUnavailable as exc:
        done.append(f"could not finish restoring: {exc}")
    clear_pending_restore()
    return done


@dataclass
class Plan:
    source: devices.Device  # where apps will play, and what we capture
    sink: devices.Device  # where you hear the transposed audio
    set_default: bool  # does the system default need changing?
    mute_source: bool  # does the source need muting?
    previous_default_id: str | None = None
    previous_source_muted: bool | None = None

    def describe(self) -> list[str]:
        lines = []
        if self.set_default:
            lines.append(f"send apps to {self.source.name!r} (changing the Windows default)")
        else:
            lines.append(f"apps already play to {self.source.name!r}")
        if self.mute_source:
            lines.append(f"mute {self.source.name!r} so you only hear the transposed version")
        lines.append(f"capture it, transpose, play to {self.sink.name!r}")
        return lines


class RoutingError(RuntimeError):
    pass


def build_plan(hear_on: str | None = None, source: str | None = None) -> Plan:
    """Work out which device plays what, preferring the user's listening device."""
    outs = devices.list_outputs()
    if len(outs) < 2:
        raise RoutingError(
            "Only one output device is active, so there is nowhere to send the "
            "transposed audio without capturing it again.\n"
            "Connect a second output (Bluetooth speaker, USB headset) or install a "
            "virtual cable, then try again.\n"
            "`autotranspose detect` needs no second device and works right now."
        )

    default = devices.default_output()
    sink = devices.resolve_output(hear_on) if hear_on else default

    if source:
        src = devices.resolve_output(source)
    else:
        src = next((d for d in outs if d.id != sink.id), None)
    if src is None or src.id == sink.id:
        raise RoutingError(
            f"The capture and playback device are both {sink.name!r}. "
            "Pass --hear-on or --source to tell them apart."
        )

    return Plan(
        source=src,
        sink=sink,
        set_default=src.id != default.id,
        mute_source=True,
    )


class Routing:
    """Applies a Plan and restores the system on exit."""

    def __init__(self, plan: Plan, enabled: bool = True):
        self.plan = plan
        self.enabled = enabled
        self.errors: list[str] = []
        self._applied = False
        self._atexit = None

    def __enter__(self) -> "Routing":
        if not self.enabled:
            return self
        p = self.plan
        # soundcard answers "what is default" reliably; the COM helper's own
        # notion of it did not match endpoint ids, and getting this wrong would
        # mean failing to put the user's default output back.
        p.previous_default_id = devices.default_output().id
        src_state = winaudio.find(p.source.id)
        p.previous_source_muted = src_state.muted if src_state else None

        save_pending_restore(
            {
                "pid": os.getpid(),
                "unmute_id": p.source.id if p.previous_source_muted is False else None,
                "unmute_name": p.source.name,
                "default_id": p.previous_default_id if p.set_default else None,
                "default_name": devices.default_output().name,
            }
        )

        if p.set_default:
            winaudio.set_default_output(p.source.id)
        if p.mute_source and not (src_state and src_state.muted):
            winaudio.set_mute(p.source.id, True)

        self._applied = True
        # Belt and braces: if the process dies without unwinding, still restore.
        self._atexit = atexit.register(self.restore)
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def restore(self) -> None:
        if not self._applied:
            return
        self._applied = False
        p = self.plan
        errors = []
        try:
            if p.previous_source_muted is False:
                winaudio.set_mute(p.source.id, False)
        except winaudio.WinAudioUnavailable as exc:
            errors.append(f"could not unmute {p.source.name}: {exc}")
        try:
            if p.set_default and p.previous_default_id:
                winaudio.set_default_output(p.previous_default_id)
        except winaudio.WinAudioUnavailable as exc:
            errors.append(f"could not restore the default output: {exc}")
        if self._atexit is not None:
            atexit.unregister(self.restore)
            self._atexit = None
        if not errors:
            clear_pending_restore()
        self.errors = errors
