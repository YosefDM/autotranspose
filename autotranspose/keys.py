"""Non-blocking keyboard reads, so the live display can take commands."""
from __future__ import annotations

import contextlib
import sys


@contextlib.contextmanager
def key_reader():
    """Yield a callable returning any keys pressed since the last call.

    Arrow keys come back as "LEFT"/"RIGHT"/"UP"/"DOWN". Falls back to a no-op
    when there is no console to read from (piped output, no TTY).
    """
    if sys.platform == "win32":
        try:
            import msvcrt
        except ImportError:  # pragma: no cover
            yield lambda: ()
            return

        def poll():
            out = []
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # extended key: the next read is the code
                    code = msvcrt.getwch() if msvcrt.kbhit() else ""
                    out.append(
                        {"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(code, "")
                    )
                else:
                    out.append(ch)
            return [c for c in out if c]

        yield poll
        return

    # POSIX
    import select
    import termios
    import tty

    if not sys.stdin.isatty():  # pragma: no cover
        yield lambda: ()
        return

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def poll():
            out = []
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    rest = ""
                    while select.select([sys.stdin], [], [], 0)[0] and len(rest) < 2:
                        rest += sys.stdin.read(1)
                    out.append(
                        {"[D": "LEFT", "[C": "RIGHT", "[A": "UP", "[B": "DOWN"}.get(rest, "")
                    )
                else:
                    out.append(ch)
            return [c for c in out if c]

        yield poll
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
