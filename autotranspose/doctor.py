"""`autotranspose doctor` -- find out why the audio sounds bad.

Runs the measurements that distinguish the causes from each other, in order:
static device facts, then capture throughput on its own, then playback on its
own, then the shifter's cost, then the whole pipeline. Each stage is reported
with a pass/fail so a failure is pinned to one stage instead of the whole app.
"""
from __future__ import annotations

import time

import numpy as np
from rich.console import Console
from rich.table import Table

from . import devices

console = Console()
SR = 48000


def _row(table, name, value, verdict, note=""):
    style = {"ok": "green", "warn": "yellow", "fail": "red", "": "dim"}[verdict]
    mark = {"ok": "ok", "warn": "warn", "fail": "FAIL", "": ""}[verdict]
    table.add_row(name, str(value), f"[{style}]{mark}[/{style}]", note)


def environment() -> None:
    import sys

    import librosa
    import sounddevice as sd
    import soundcard as sc

    t = Table(title="Environment", show_header=False, box=None, padding=(0, 2))
    t.add_row("python", sys.version.split()[0])
    t.add_row("numpy", np.__version__)
    t.add_row("librosa", librosa.__version__)
    t.add_row("sounddevice", sd.__version__)
    t.add_row("soundcard", getattr(sc, "__version__", "?"))
    console.print(t)
    console.print()

    t = Table(title="Output devices (playback path)", box=None, padding=(0, 2))
    t.add_column("idx")
    t.add_column("name")
    t.add_column("host API")
    t.add_column("ch")
    t.add_column("default sr")
    t.add_column("low latency")
    wasapi = [i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"]]
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] <= 0 or (wasapi and d["hostapi"] not in wasapi):
            continue
        t.add_row(
            str(i),
            d["name"],
            sd.query_hostapis(d["hostapi"])["name"],
            str(d["max_output_channels"]),
            str(int(d["default_samplerate"])),
            f"{d['default_low_output_latency'] * 1000:.1f} ms",
        )
    console.print(t)
    console.print()


def capture_throughput(capture: devices.Device, seconds: float = 3.0) -> dict:
    """Can we pull loopback audio at real time? Anything over 1.0x is trouble."""
    blocksize = 1024
    n = int(seconds * SR / blocksize)
    worst = 0.0
    with devices.open_recorder(capture, SR, 2, blocksize) as rec:
        rec.record(numframes=blocksize)
        t0 = time.perf_counter()
        for _ in range(n):
            t1 = time.perf_counter()
            rec.record(numframes=blocksize)
            worst = max(worst, time.perf_counter() - t1)
        wall = time.perf_counter() - t0
    audio = n * blocksize / SR
    return {"ratio": wall / audio, "worst_ms": worst * 1000, "audio_s": audio}


def playback_throughput(output: devices.Device, seconds: float = 3.0) -> dict:
    """Does the output stream keep up without underflowing?"""
    import sounddevice as sd

    state = {"calls": 0, "under": 0, "phase": 0.0, "worst": 0.0, "last": 0.0}

    def cb(outdata, frames, time_info, status):
        t0 = time.perf_counter()
        state["calls"] += 1
        if status and getattr(status, "output_underflow", False):
            state["under"] += 1
        k = np.arange(frames)
        x = (0.05 * np.sin(2 * np.pi * 220 * k / SR + state["phase"])).astype(np.float32)
        state["phase"] = (state["phase"] + 2 * np.pi * 220 * frames / SR) % (2 * np.pi)
        outdata[:, 0] = x
        if outdata.shape[1] > 1:
            outdata[:, 1] = x
        state["worst"] = max(state["worst"], time.perf_counter() - t0)

    with sd.OutputStream(
        device=devices.sd_output_index(output),
        samplerate=SR,
        channels=2,
        blocksize=1024,
        dtype="float32",
        callback=cb,
    ) as stream:
        latency = stream.latency
        time.sleep(seconds)
    expected = seconds * SR / 1024
    return {
        "calls": state["calls"],
        "expected": expected,
        "underflows": state["under"],
        "latency_ms": latency * 1000,
        "worst_cb_ms": state["worst"] * 1000,
    }


def shifter_cost(n_fft: int = 2048, crossfade: bool = True) -> dict:
    from .shifter import _PhaseVocoderCore

    blocksize = 1024
    cores = [
        _PhaseVocoderCore(2, n_fft, 4, 2 ** (5 / 12.0)) for _ in range(2 if crossfade else 1)
    ]
    x = (np.random.default_rng(0).standard_normal((2, blocksize)) * 0.1).astype(np.float32)
    for _ in range(30):
        for c in cores:
            c.process(x)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        for c in cores:
            c.process(x)
    per_block = (time.perf_counter() - t0) / n
    budget = blocksize / SR
    return {"per_block_ms": per_block * 1000, "budget_ms": budget * 1000,
            "load_pct": per_block / budget * 100}


def pipeline(seconds: float, args) -> tuple[dict, list[str], object]:
    """Run the real thing and report its own diagnostics."""
    from . import autoroute
    from .cli import _config, _diagnostics
    from .engine import Engine

    plan = autoroute.build_plan(hear_on=getattr(args, "hear_on", None))
    cfg = _config(args)
    diag = _diagnostics(args, cfg)
    capture = devices.resolve_capture(plan.source.name)
    engine = Engine(capture, plan.sink, cfg, diagnostics=diag)
    console.print(
        f"[dim]Running the full pipeline for {seconds:.0f}s: "
        f"{capture.name} -> {plan.sink.name}[/dim]"
    )
    console.print("[dim]Play something now if you can; silence tests less.[/dim]")
    with autoroute.Routing(plan):
        with engine:
            time.sleep(seconds)
    stats = diag.summarise(final=True)
    return stats, diag.verdict(stats), diag


def run(args) -> int:
    seconds = getattr(args, "seconds", 0) or 12.0
    environment()

    try:
        capture = devices.resolve_capture(getattr(args, "capture", "default"))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    t = Table(title="Measurements", box=None, padding=(0, 2))
    t.add_column("stage")
    t.add_column("result")
    t.add_column("")
    t.add_column("note")

    # 1. capture
    try:
        cap = capture_throughput(capture)
        verdict = "ok" if cap["ratio"] < 1.05 else ("warn" if cap["ratio"] < 1.2 else "fail")
        _row(
            t, f"capture ({capture.name})", f"{cap['ratio']:.2f}x real time", verdict,
            f"worst block {cap['worst_ms']:.0f} ms; over 1.05x starves the pipeline",
        )
    except Exception as exc:
        _row(t, "capture", f"{type(exc).__name__}", "fail", str(exc)[:60])

    # 2. playback, per device
    outs = devices.list_outputs()
    for dev in outs:
        try:
            pb = playback_throughput(dev, seconds=2.0)
            short = pb["calls"] < pb["expected"] * 0.9
            verdict = "fail" if (pb["underflows"] or short) else "ok"
            _row(
                t, f"playback ({dev.name})",
                f"{pb['calls']}/{pb['expected']:.0f} callbacks, {pb['underflows']} underflows",
                verdict,
                f"stream latency {pb['latency_ms']:.0f} ms, worst callback "
                f"{pb['worst_cb_ms']:.2f} ms",
            )
        except Exception as exc:
            _row(t, f"playback ({dev.name})", type(exc).__name__, "fail", str(exc)[:60])

    # 3. shifter cost
    for n_fft in (1024, 2048, 4096):
        sc_ = shifter_cost(n_fft, crossfade=not getattr(args, "no_crossfade", False))
        verdict = "ok" if sc_["load_pct"] < 40 else ("warn" if sc_["load_pct"] < 70 else "fail")
        _row(
            t, f"shifter fft={n_fft}", f"{sc_['load_pct']:.1f}% of one core", verdict,
            f"{sc_['per_block_ms']:.2f} ms per {sc_['budget_ms']:.1f} ms block",
        )

    console.print(t)
    console.print()

    if len(outs) < 2 and not getattr(args, "no_pipeline", False):
        console.print(
            "[yellow]Only one output device, so the full pipeline cannot be tested.[/yellow]"
        )
        return 0

    if getattr(args, "no_pipeline", False):
        return 0

    try:
        stats, verdict, diag = pipeline(seconds, args)
    except Exception as exc:
        console.print(f"[red]pipeline test failed: {type(exc).__name__}: {exc}[/red]")
        return 1

    t = Table(title="Full pipeline", show_header=False, box=None, padding=(0, 2))
    for key in (
        "elapsed_s", "blocks", "process_ms", "load_pct", "interval_ms", "jitter_ms",
        "ring_frames", "ring_ms", "in_peak", "out_peak", "gain_db", "clipped",
        "drift_frames", "drift_ms", "drift_ppm", "cb_shortfall_frames", "events",
    ):
        if key in stats:
            t.add_row(key, str(stats[key]))
    console.print(t)
    console.print()

    console.print("[bold]Verdict[/bold]")
    for line in verdict:
        style = "green" if line.startswith("nothing wrong") else "yellow"
        console.print(f"  [{style}]-[/{style}] {line}")

    events = diag.recent_events(15)
    if events:
        console.print()
        console.print("[bold]Last events[/bold]")
        for name, t_rel in events:
            console.print(f"  [dim]{t_rel:6.2f}s[/dim] {name}")

    if diag.log_path:
        console.print()
        console.print(f"[dim]full log: {diag.log_path}[/dim]")
    return 0
