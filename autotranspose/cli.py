"""Command line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import devices
from .engine import Engine, EngineConfig

console = Console()


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--capture",
        default="default",
        help="capture source: index, name substring, or 'default' "
        "(loopback of the current output device)",
    )
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--blocksize", type=int, default=1024, help="audio block size in frames")
    p.add_argument(
        "--objective",
        default="white",
        choices=["white", "template"],
        help="'white' picks the shift that puts most of the music on white keys "
        "(default, stabler); 'template' is the older key-profile method",
    )
    p.add_argument(
        "--min-gain",
        type=float,
        default=5.0,
        help="percentage points of white-key coverage a change must buy before it "
        "happens (default 3). Higher = more stubborn",
    )
    p.add_argument(
        "--profile",
        default="blend",
        choices=["blend", "shaath", "temperley", "krumhansl", "albrecht"],
        help="key profile used for detection (default: blend of three)",
    )
    p.add_argument(
        "--window", type=float, default=30.0, help="seconds of audio the detector averages over"
    )
    p.add_argument(
        "--min-heard",
        type=float,
        default=12.0,
        help="seconds of music required before the first lock (default 12)",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=25.0,
        help="minimum seconds between shift changes (default 12)",
    )
    p.add_argument(
        "--commit-share",
        type=float,
        default=0.55,
        help="fraction of the vote a shift needs before it is applied",
    )
    p.add_argument(
        "--seconds", type=float, default=0.0, help="stop after N seconds (0 = run until quit)"
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="log one line per change instead of the live panel; good for "
        "checking detection against your own library",
    )
    g = p.add_argument_group("diagnostics")
    g.add_argument(
        "--log-file",
        default=None,
        help="where to write the session log (default: "
        "%%LOCALAPPDATA%%\\autotranspose\\logs\\session-<time>.log); "
        "'none' disables it",
    )
    g.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="DEBUG adds a line per analysis cycle",
    )
    g.add_argument("--log-console", action="store_true", help="also print the log to the terminal")
    g.add_argument(
        "--summary-every",
        type=float,
        default=5.0,
        help="seconds between timing/dropout summaries in the log",
    )
    g.add_argument(
        "--diag-csv",
        default=None,
        help="write per-block metrics to this CSV on exit, for plotting",
    )
    g.add_argument(
        "--record-wav",
        default=None,
        help="directory to write input.wav and output.wav, to compare quality by ear",
    )


def _restore_if_pending() -> None:
    """If a previous run was killed before cleaning up, put things back."""
    from . import autoroute

    if autoroute.pending_restore() is None:
        return
    other = autoroute.owned_by_live_process()
    if other:
        console.print(
            f"[yellow]Another autotranspose is already running (pid {other}) and has "
            f"your audio settings changed. Leaving them alone.[/yellow]\n"
        )
        return
    done = autoroute.restore_pending()
    if done:
        console.print("[yellow]A previous run did not clean up. Restored:[/yellow]")
        for line in done:
            console.print(f"  [dim]-[/dim] {line}")
        console.print()


def cmd_restore(args) -> int:
    from . import autoroute

    done = autoroute.restore_pending(force=getattr(args, "force", False))
    if not done:
        console.print("Nothing to restore; no interrupted run is recorded.")
        return 0
    for line in done:
        console.print(f"  [dim]-[/dim] {line}")
    return 0


def cmd_devices(args) -> int:
    console.print("[bold]Capture sources[/bold]  (use with --capture)")
    for i, d in enumerate(devices.list_capture_sources()):
        tag = "[green]loopback[/green]" if d.is_loopback else "[dim]microphone[/dim]"
        console.print(f"  [{i}] {d.name}  {tag}  [dim]{d.channels}ch[/dim]")
    console.print()
    console.print("[bold]Output devices[/bold]  (use with --output)")
    for i, d in enumerate(devices.list_outputs()):
        console.print(f"  [{i}] {d.name}  [dim]{d.channels}ch[/dim]")
    console.print()
    cap, out = devices.default_capture(), devices.default_output()
    console.print(f"[dim]default capture:[/dim] {cap.name}")
    console.print(f"[dim]default output: [/dim] {out.name}")
    chk = devices.check_routing(cap, out)
    if not chk.ok:
        console.print()
        console.print(f"[yellow]Note:[/yellow] {chk.message}")
        console.print("[dim]Run `autotranspose detect` to analyse without playback.[/dim]")
    return 0


def _config(args) -> EngineConfig:
    return EngineConfig(
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        n_fft=getattr(args, "fft", 2048),
        crossfade=not getattr(args, "no_crossfade", False),
        analysis_window=args.window,
        profile=args.profile,
        objective=getattr(args, "objective", "white"),
        min_gain=getattr(args, "min_gain", 3.0) / 100.0,
        max_shift=getattr(args, "max_shift", 6),
        prefer_down=not getattr(args, "prefer_up", False),
        min_heard_seconds=args.min_heard,
        min_dwell_seconds=args.dwell,
        commit_share=args.commit_share,
        mono=getattr(args, "mono", False),
        bypass=getattr(args, "bypass", False),
        shift_engine=getattr(args, "engine", "auto"),
        output_gain=getattr(args, "output_gain", 1.0),
        gc_freeze=getattr(args, "gc_freeze", False),
        record_dir=getattr(args, "record_wav", None),
        ring_blocks=getattr(args, "ring_blocks", 12),
    )


def _diagnostics(args, cfg):
    """Build the Diagnostics for this run from the command line."""
    from .diag import Diagnostics, default_log_path

    spec = getattr(args, "log_file", None)
    if spec and spec.lower() == "none":
        path = None
    else:
        path = Path(spec) if spec else default_log_path()
    csv = getattr(args, "diag_csv", None)
    return Diagnostics(
        samplerate=cfg.samplerate,
        blocksize=cfg.blocksize,
        log_path=path,
        level=getattr(args, "log_level", "INFO"),
        console=getattr(args, "log_console", False),
        summary_seconds=getattr(args, "summary_every", 5.0),
        csv_path=Path(csv) if csv else None,
    )


def _build(args, output) -> Engine:
    cfg = _config(args)
    return Engine(
        devices.resolve_capture(args.capture), output, cfg, diagnostics=_diagnostics(args, cfg)
    )


def _run(engine: Engine, args) -> int:
    from .tui import run_live, run_plain

    chk = devices.check_routing(engine.capture, engine.output)
    if not chk.ok:
        console.print(f"[bold red]{chk.message}[/bold red]\n")
        console.print(chk.advice)
        return 2
    console.print(f"[dim]{chk.message}[/dim]")

    seconds = getattr(args, "seconds", 0.0) or 0.0
    plain = getattr(args, "plain", False) or not sys.stdout.isatty()
    if engine.diag.log_path:
        console.print(f"[dim]log: {engine.diag.log_path}[/dim]")
    try:
        with engine:
            if plain:
                run_plain(engine, seconds=seconds)
            else:
                run_live(engine, seconds=seconds)
    except KeyboardInterrupt:
        pass

    verdict = engine.diag.verdict()
    if verdict:
        console.print()
        console.print("[bold]Audio diagnostics[/bold]")
        for line in verdict:
            style = "green" if line.startswith("nothing wrong") else "yellow"
            console.print(f"  [{style}]-[/{style}] {line}")
    if engine.diag.log_path:
        console.print(f"[dim]full log: {engine.diag.log_path}[/dim]")
    if engine.status.error:
        console.print(f"[bold red]error:[/bold red] {engine.status.error}")
        return 1
    return 0


def cmd_detect(args) -> int:
    engine = _build(args, None)
    console.print(
        "[bold]Analyse only[/bold] - detecting the key and the shift you would need. "
        "Nothing is played back.\n"
    )
    return _run(engine, args)


def cmd_run(args) -> int:
    from . import autoroute, winaudio

    # Explicit devices mean the user is doing their own routing (a virtual
    # cable, say); don't touch the system in that case.
    manual = args.capture != "default" or args.output not in (None, "", "auto")
    if manual or args.no_auto_route:
        capture = devices.resolve_capture(args.capture)
        output = devices.resolve_output("default" if args.output == "auto" else args.output)
        cfg = _config(args)
        return _run(
            Engine(capture, output, cfg, diagnostics=_diagnostics(args, cfg)), args
        )

    try:
        plan = autoroute.build_plan(hear_on=args.hear_on, source=args.source)
    except autoroute.RoutingError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    if not winaudio.available():
        console.print(
            "[yellow]Cannot drive the Windows audio settings[/yellow] "
            "(pycaw/comtypes unavailable), so the routing cannot be set up "
            "automatically. Install them with `pip install -r requirements.txt`, "
            "or set it up by hand and pass --capture/--output.\n"
        )
        return 2

    console.print("[bold]Setting up:[/bold]")
    for line in plan.describe():
        console.print(f"  [dim]-[/dim] {line}")
    console.print("[dim]Everything is put back when you quit.[/dim]\n")

    capture = devices.resolve_capture(plan.source.name)
    cfg = _config(args)
    engine = Engine(capture, plan.sink, cfg, diagnostics=_diagnostics(args, cfg))
    routing = autoroute.Routing(plan)
    try:
        with routing:
            code = _run(engine, args)
    finally:
        routing.restore()
        for err in routing.errors:
            console.print(f"[yellow]warning:[/yellow] {err}")
        if not routing.errors:
            console.print("[dim]Audio settings restored.[/dim]")
    return code


def cmd_doctor(args) -> int:
    from . import doctor

    return doctor.run(args)


def cmd_analyse(args) -> int:
    """Compare the recorded input and output WAVs."""
    from .record import compare

    d = Path(args.directory)
    res = compare(d / "input.wav", d / "output.wav")
    for k, v in res.items():
        console.print(f"  {k:24s} {v}")
    console.print()
    if res.get("dropout_windows_10ms"):
        console.print(
            f"[yellow]{res['dropout_windows_10ms']} ten-millisecond windows were loud on "
            f"input but silent on output -- audible dropouts.[/yellow]"
        )
    if res.get("clipped_out"):
        console.print(
            f"[yellow]{res['clipped_out']} clipped output samples; try "
            f"--output-gain 0.8.[/yellow]"
        )
    if not res.get("dropout_windows_10ms") and not res.get("clipped_out"):
        console.print("[green]No dropouts and no clipping in the recording.[/green]")
    return 0


def cmd_check(args) -> int:
    """Offline sanity check on a file, for comparing against a known key."""
    import numpy as np
    import soundfile as sf

    from .keydetect import ANALYSIS_SR, detect_key_offline
    from .keyprofiles import signed_shift

    y, sr = sf.read(args.path, dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    if args.seconds:
        mono = mono[: int(args.seconds * sr)]
    est = detect_key_offline(mono, sr, profile=args.profile)
    shift = signed_shift(est.shift_class)
    console.print(f"file:       {args.path}")
    console.print(f"duration:   {len(mono) / sr:.1f}s")
    console.print(f"key:        [bold]{est.key_name}[/bold]  (confidence {est.key_confidence:.1%})")
    console.print(f"shift:      [bold cyan]{shift:+d} semitones[/bold cyan] to reach C major / A minor")
    console.print(f"margin:     {est.margin:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autotranspose",
        description="Detect the key of whatever your computer is playing and "
        "transpose it onto the white keys, live.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("devices", help="list capture sources and output devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser(
        "detect",
        help="listen and show the key + required shift, without playing anything "
        "(needs no audio routing)",
    )
    _add_common(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("run", help="capture, transpose and play back to your headphones")
    _add_common(p)
    p.add_argument(
        "--output",
        default="auto",
        help="device you hear the transposed audio on; 'auto' arranges everything for you",
    )
    p.add_argument(
        "--hear-on",
        default=None,
        help="name of the device to listen on (default: whatever Windows is set to now)",
    )
    p.add_argument(
        "--source",
        default=None,
        help="name of the device apps should play into and we capture "
        "(default: the other active output)",
    )
    p.add_argument(
        "--no-auto-route",
        action="store_true",
        help="do not touch the Windows default output or mute anything; "
        "set the routing up yourself",
    )
    p.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "signalsmith", "vocoder"],
        help="pitch-shift engine: 'signalsmith' is the native DLL (better and "
        "cheaper), 'vocoder' the pure-Python fallback, 'auto' prefers Signalsmith",
    )
    p.add_argument("--fft", type=int, default=2048, help="fallback vocoder window (1024-4096)")
    p.add_argument(
        "--no-crossfade",
        action="store_true",
        help="switch shift instantly instead of crossfading; halves CPU",
    )
    p.add_argument("--mono", action="store_true", help="collapse to mono before shifting")
    p.add_argument(
        "--max-shift", type=int, default=6, help="largest shift in semitones (default 6)"
    )
    p.add_argument(
        "--prefer-up",
        action="store_true",
        help="shift up rather than down when both are equally far",
    )
    p.add_argument(
        "--bypass", action="store_true", help="pass audio through unshifted (to A/B the quality)"
    )
    p.add_argument(
        "--output-gain",
        type=float,
        default=1.0,
        help="scale the output; use 0.8 if the log reports clipping",
    )
    p.add_argument(
        "--gc-freeze",
        action="store_true",
        help="freeze and disable the garbage collector, if gc pauses cause dropouts",
    )
    p.add_argument(
        "--ring-blocks",
        type=int,
        default=12,
        help="playback buffer depth in blocks (default 12 = ~256 ms). More means "
        "fewer dropouts and more latency",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "restore",
        help="undo audio-setting changes left behind by an interrupted run",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="restore even if another autotranspose appears to be running",
    )
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser(
        "doctor",
        help="diagnose choppy or poor-quality audio: measures capture, playback, "
        "the shifter and the whole pipeline, then says what is wrong",
    )
    _add_common(p)
    p.add_argument(
        "--hear-on", default=None, help="device to listen on during the pipeline test"
    )
    p.add_argument("--fft", type=int, default=2048)
    p.add_argument("--no-crossfade", action="store_true")
    p.add_argument("--mono", action="store_true")
    p.add_argument("--output-gain", type=float, default=1.0)
    p.add_argument("--engine", default="auto", choices=["auto", "signalsmith", "vocoder"])
    p.add_argument("--gc-freeze", action="store_true")
    p.add_argument("--ring-blocks", type=int, default=12)
    p.add_argument(
        "--no-pipeline",
        action="store_true",
        help="only run the static and per-stage measurements, nothing audible",
    )
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser(
        "analyse",
        help="compare input.wav and output.wav from a --record-wav run",
    )
    p.add_argument("directory")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("check", help="detect the key of an audio file (offline sanity check)")
    p.add_argument("path")
    p.add_argument("--seconds", type=float, default=0, help="only analyse the first N seconds")
    p.add_argument("--profile", default="blend")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    if args.command != "restore":
        _restore_if_pending()
    try:
        return args.func(args)
    except ValueError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
