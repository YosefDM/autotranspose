"""Live terminal display."""
from __future__ import annotations

import time

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .decide import shift_explanation
from .engine import Engine
from .keyprofiles import PITCH_CLASS_NAMES, key_indices_for_shift_class, key_label

STATUS_STYLE = {
    "locked": "bold green",
    "listening": "bold yellow",
    "silent": "dim",
    "manual": "bold magenta",
    "starting": "dim",
}


def _meter(dbfs: float, width: int = 24) -> Text:
    frac = max(0.0, min(1.0, (dbfs + 60.0) / 60.0))
    filled = int(frac * width)
    colour = "green" if dbfs < -12 else ("yellow" if dbfs < -3 else "red")
    bar = Text("#" * filled, style=colour)
    bar.append("." * (width - filled), style="dim")
    bar.append(f"  {dbfs:6.1f} dB" if dbfs > -99 else "   --   ", style="dim")
    return bar


def _shift_text(shift: int) -> Text:
    if shift == 0:
        return Text("0 (no shift)", style="bold white")
    arrow = "down" if shift < 0 else "up"
    return Text(f"{shift:+d} semitones ({arrow})", style="bold cyan")


def _votes_table(votes, leader: int | None) -> Table:
    t = Table.grid(padding=(0, 1))
    t.add_row(*[Text("shift", style="dim")] + [
        Text(f"{s:+d}" if s else "0", style="dim")
        for s in [_signed(i) for i in range(12)]
    ])
    cells = [Text("vote", style="dim")]
    for i in range(12):
        share = float(votes[i]) if len(votes) == 12 else 0.0
        style = "bold green" if i == leader else ("white" if share > 0.1 else "dim")
        cells.append(Text(f"{share * 100:3.0f}", style=style))
    t.add_row(*cells)
    return t


SILENT_HINT_AFTER = 6.0


def _silent_too_long(engine) -> bool:
    st = engine.status
    if st.decision.status not in ("silent", "starting"):
        return False
    return st.started_at and (time.monotonic() - st.started_at) > SILENT_HINT_AFTER


def _signed(shift_class: int) -> int:
    up = shift_class % 12
    down = up - 12
    return up if abs(up) < abs(down) else down


def render(engine: Engine) -> Panel:
    st = engine.status
    d = st.decision
    style = STATUS_STYLE.get(d.status, "white")

    head = Table.grid(padding=(0, 2))
    head.add_column(justify="right", style="dim")
    head.add_column()

    status_line = Text(d.status.upper(), style=style)
    if d.status == "listening":
        need = engine.decider.min_heard_seconds
        status_line.append(f"   heard {d.heard_seconds:4.1f}s / {need:.0f}s needed", style="dim")
    elif d.status == "locked" and d.locked_at:
        status_line.append(f"   stable for {time.monotonic() - d.locked_at:4.0f}s", style="dim")
    head.add_row("status", status_line)

    head.add_row("playing in", Text(d.key_name if d.key_name != "--" else "listening...",
                                    style="bold" if d.key_name != "--" else "dim"))
    head.add_row("transpose", _shift_text(st.applied_shift))

    if d.candidate_class is not None:
        maj, minr = key_indices_for_shift_class(d.candidate_class)
        head.add_row(
            "white keys",
            Text("play in C major / A minor", style="green")
            if st.applied_shift == _signed(d.candidate_class)
            else Text(
                f"pending {_signed(d.candidate_class):+d} "
                f"({key_label(maj)} / {key_label(minr)})",
                style="yellow",
            ),
        )
    if d.white_best > 0:
        gain = (d.white_best - d.white_now) * 100
        txt = Text(f"{d.white_now * 100:.0f}% of the notes", style="bold green")
        if gain > 0.5:
            txt.append(
                f"   (best available {d.white_best * 100:.0f}%, +{gain:.0f})", style="yellow"
            )
        head.add_row("on white keys", txt)
    head.add_row("input", _meter(st.input_dbfs))
    if engine.output is not None:
        head.add_row("output", _meter(st.output_dbfs))

    detail = Table.grid(padding=(0, 2))
    detail.add_column(justify="right", style="dim")
    detail.add_column()
    detail.add_row(
        "confidence",
        Text(f"{d.key_confidence * 100:5.1f}%  leader share {d.leader_share * 100:5.1f}%  "
             f"margin {d.margin:.3f}", style="dim"),
    )
    ring_ms = st.ring_frames / max(engine.cfg.samplerate, 1) * 1000.0
    latency = st.shifter_latency_ms + st.block_latency_ms + ring_ms
    detail.add_row(
        "timing",
        Text(
            f"~{latency:.0f} ms added latency   "
            f"audio {st.audio_load * 100:4.1f}% of one block   "
            f"analysis {st.analysis_ms:5.1f} ms/cycle",
            style="dim",
        ),
    )
    health = []
    if st.underruns:
        health.append(f"{st.underruns} underruns")
    if st.overruns:
        health.append(f"{st.overruns} drops")
    if st.discontinuities:
        health.append(f"{st.discontinuities} capture gaps")
    if st.clipped_samples:
        health.append(f"{st.clipped_samples} clipped")
    detail.add_row(
        "audio health",
        Text(
            "  ".join(health) if health else "clean - no dropouts, no clipping",
            style="yellow" if health else "green",
        ),
    )
    detail.add_row(
        "peaks",
        Text(
            f"in {st.peak_in:.3f}  out {st.peak_out:.3f}"
            + ("   CLIPPING" if st.peak_out >= 0.999 else ""),
            style="red" if st.peak_out >= 0.999 else "dim",
        ),
    )

    extras = []
    if d.changes:
        extras.append(f"{d.changes} key change{'s' if d.changes != 1 else ''}")
    if st.dropped_analysis_blocks:
        extras.append(f"{st.dropped_analysis_blocks} analysis blocks dropped")
    if st.handovers:
        extras.append(f"{st.handovers} crossfades")
    if extras:
        detail.add_row("session", Text("   ".join(extras), style="dim"))

    body = [head, Text(""), _votes_table(d.votes, d.candidate_class), Text(""), detail]

    if _silent_too_long(engine):
        body += [
            Text(""),
            Text(
                f"No audio arriving on {engine.capture.name!r}.\n"
                "If your music was already playing when this started, pause and\n"
                "resume it -- apps with an open stream do not always follow a\n"
                "change of default output device.",
                style="yellow",
            ),
        ]

    if st.error:
        body += [Text(""), Text(f"error: {st.error}", style="bold red")]

    keys = Text(
        "  [q] quit   [a] auto/manual   [<-/->] nudge shift   [0] no shift",
        style="dim",
    )
    body += [Text(""), keys]

    title = "autotranspose"
    sub = (
        f"{engine.capture.name}  ->  {engine.output.name}"
        if engine.output
        else f"{engine.capture.name}  (analyse only)"
    )
    return Panel(
        Group(*body),
        title=title,
        subtitle=sub,
        border_style=style if d.status != "starting" else "dim",
        padding=(1, 2),
    )


def run_live(engine: Engine, refresh: float = 10.0, seconds: float = 0.0) -> None:
    """Draw the display until the engine stops, time runs out, or the user quits."""
    from .keys import key_reader

    deadline = time.monotonic() + seconds if seconds else None
    with Live(render(engine), refresh_per_second=refresh, screen=False) as live:
        with key_reader() as keys:
            while engine.status.running and not engine.status.error:
                if deadline and time.monotonic() >= deadline:
                    break
                for ch in keys():
                    if ch in ("q", "\x03"):
                        return
                    if ch == "a":
                        engine.toggle_auto()
                    elif ch == "0":
                        engine.set_manual(0)
                    elif ch == "LEFT":
                        engine.nudge(-1)
                    elif ch == "RIGHT":
                        engine.nudge(+1)
                live.update(render(engine))
                time.sleep(1.0 / refresh)
            if engine.status.error:
                live.update(render(engine))


def run_plain(engine: Engine, seconds: float = 0.0, interval: float = 1.0) -> None:
    """One line per cycle, for logging detection against your own library.

    Useful where the live panel will not render (piped output, no TTY) and for
    checking how the detector behaves on real music over a whole album.
    """
    from rich.console import Console

    console = Console()
    deadline = time.monotonic() + seconds if seconds else None
    started = time.monotonic()
    last_shift = None
    last_status = None
    hinted = False
    last_print = 0.0
    HEARTBEAT = 5.0
    try:
        while engine.status.running and not engine.status.error:
            if deadline and time.monotonic() >= deadline:
                break
            st = engine.status
            d = st.decision
            now = time.monotonic()
            changed = d.status != last_status or st.applied_shift != last_shift
            if changed or now - last_print >= HEARTBEAT:
                last_print = now
                marker = "*" if st.applied_shift != last_shift and last_shift is not None else " "
                console.print(
                    "%s %6.1fs  %-9s  in %6.1f dB  key %-9s  shift %+d  "
                    "share %3.0f%%  conf %3.0f%%"
                    % (
                        marker,
                        time.monotonic() - started,
                        d.status,
                        st.input_dbfs,
                        d.key_name,
                        st.applied_shift,
                        d.leader_share * 100,
                        d.key_confidence * 100,
                    )
                )
                last_status, last_shift = d.status, st.applied_shift
            if not hinted and _silent_too_long(engine):
                hinted = True
                console.print(
                    "[yellow]No audio arriving on %r. If your music was already "
                    "playing, pause and resume it so it follows the new default "
                    "output.[/yellow]" % engine.capture.name
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    if engine.status.error:
        console.print(f"[bold red]error:[/bold red] {engine.status.error}")
