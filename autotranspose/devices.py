"""Audio device discovery and routing checks.

Capture uses WASAPI loopback, which records whatever a render device is already
playing -- no virtual cable needed to *listen*. The catch is playback: if we play
the shifted audio back to the same device we are capturing from, our own output
is captured and shifted again, and again. `check_routing` is what stops that.
"""
from __future__ import annotations

from dataclasses import dataclass

import soundcard as sc


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    channels: int
    is_loopback: bool = False

    def __str__(self) -> str:
        return self.name


def list_capture_sources() -> list[Device]:
    """Loopback devices (what the computer is playing) plus real microphones."""
    out = []
    for m in sc.all_microphones(include_loopback=True):
        out.append(Device(str(m.id), m.name, m.channels, bool(m.isloopback)))
    return out


def list_outputs() -> list[Device]:
    return [Device(str(s.id), s.name, s.channels) for s in sc.all_speakers()]


def default_capture() -> Device:
    """Loopback of the default output -- i.e. 'whatever I am listening to'."""
    spk = sc.default_speaker()
    mic = sc.get_microphone(spk.name, include_loopback=True)
    return Device(str(mic.id), mic.name, mic.channels, True)


def default_output() -> Device:
    s = sc.default_speaker()
    return Device(str(s.id), s.name, s.channels)


def resolve_capture(spec: str | None) -> Device:
    if spec in (None, "", "default"):
        return default_capture()
    sources = list_capture_sources()
    return _match(spec, sources, "capture source")


def resolve_output(spec: str | None) -> Device:
    if spec in (None, "", "default"):
        return default_output()
    return _match(spec, list_outputs(), "output device")


def _match(spec: str, devices: list[Device], what: str) -> Device:
    if spec.isdigit():
        i = int(spec)
        if not 0 <= i < len(devices):
            raise ValueError(f"{what} index {i} out of range (0-{len(devices) - 1})")
        return devices[i]
    low = spec.lower()
    hits = [d for d in devices if low in d.name.lower()]
    if not hits:
        names = "\n  ".join(f"[{i}] {d.name}" for i, d in enumerate(devices))
        raise ValueError(f"no {what} matching {spec!r}. Available:\n  {names}")
    if len(hits) > 1:
        exact = [d for d in hits if d.name.lower() == low]
        if len(exact) == 1:
            return exact[0]
    return hits[0]


def open_recorder(device: Device, samplerate: int, channels: int, blocksize: int):
    mic = sc.get_microphone(device.id, include_loopback=True)
    return mic.recorder(samplerate=samplerate, channels=channels, blocksize=blocksize)


def sd_output_index(device: Device) -> int:
    """Find the sounddevice/PortAudio index for a WASAPI output.

    Playback goes through sounddevice rather than soundcard: soundcard's
    `Player.play()` costs a fixed ~9 ms per call on this machine, so feeding it
    one capture block at a time runs at 1.4x real time and the device starves.
    sounddevice's callback-driven WASAPI output sustains real time with zero
    underflows. Capture still uses soundcard, which is the only one of the two
    that exposes WASAPI loopback here.
    """
    import sounddevice as sd

    wasapi = [i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"]]
    candidates = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_output_channels"] > 0 and (not wasapi or d["hostapi"] in wasapi)
    ]
    target = device.name.strip().lower()
    for i, d in candidates:
        if d["name"].strip().lower() == target:
            return i
    # PortAudio sometimes truncates long device names.
    for i, d in candidates:
        n = d["name"].strip().lower()
        if n and (n in target or target in n):
            return i
    available = "\n  ".join(f"[{i}] {d['name']}" for i, d in candidates)
    raise ValueError(
        f"could not find {device.name!r} among the playback devices:\n  {available}"
    )


def _base_name(name: str) -> str:
    return name.lower().replace("loopback", "").strip()


@dataclass
class RoutingCheck:
    ok: bool
    severity: str  # "ok" | "warn" | "error"
    message: str
    advice: str = ""


def check_routing(capture: Device, output: Device | None) -> RoutingCheck:
    """Catch the feedback loop before it deafens anyone."""
    if output is None:
        return RoutingCheck(True, "ok", "Analyse-only: nothing is played back.")

    if not capture.is_loopback:
        return RoutingCheck(
            True,
            "ok",
            f"Capturing microphone {capture.name!r} (not loopback) -> {output.name!r}.",
        )

    if _base_name(capture.name) == _base_name(output.name):
        return RoutingCheck(
            False,
            "error",
            f"Feedback loop: capturing the loopback of {output.name!r} while also "
            f"playing back into it. The shifted audio would be re-captured and "
            f"re-shifted until it howls.",
            advice=(
                "Pick different devices. Either:\n"
                "  1. Route the music app to a virtual cable and capture that "
                "(--capture 'CABLE Output'), playing back to your headphones, or\n"
                "  2. Play back to a separate output (USB/Bluetooth headphones) "
                "with --output.\n"
                "Run `autotranspose devices` to see what is available, and "
                "`autotranspose detect` to analyse without playback."
            ),
        )
    return RoutingCheck(
        True, "ok", f"Capturing loopback of {capture.name!r} -> playing to {output.name!r}."
    )
