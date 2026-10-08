"""Real end-to-end test: play music through the speakers, let the app hear it.

This is the only test that uses the actual sound card. It plays a synthetic song
in a known key out of the default output while the engine captures that device's
WASAPI loopback, and checks the engine locks onto the right shift.

Audible. Run it when you can hear the speakers, and skip it in CI.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundcard as sc  # noqa: E402
from synth import synth_song  # noqa: E402

from autotranspose import devices  # noqa: E402
from autotranspose.engine import Engine, EngineConfig  # noqa: E402
from autotranspose.keyprofiles import shift_class_for_key, signed_shift  # noqa: E402

SR = 48000


def play_song(song, stop_event, volume=0.35):
    spk = sc.default_speaker()
    stereo = np.stack([song, song], axis=1) * volume
    with spk.player(samplerate=SR, channels=2, blocksize=2048) as p:
        while not stop_event.is_set():
            p.play(stereo)


def run_case(tonic_pc, minor, label, timeout=40.0):
    want_shift = signed_shift(shift_class_for_key(tonic_pc + (12 if minor else 0)))
    song = synth_song(tonic_pc, minor_key=minor, duration=20.0, sr=SR, seed=42)

    stop = threading.Event()
    t = threading.Thread(target=play_song, args=(song, stop), daemon=True)
    t.start()
    time.sleep(1.0)  # let the output device spin up before we start listening

    cfg = EngineConfig(samplerate=SR, blocksize=1024, min_heard_seconds=10.0)
    eng = Engine(devices.default_capture(), None, cfg)  # analyse only: no playback

    print("\n%s (expecting %+d):" % (label, want_shift))
    locked = None
    try:
        with eng:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < timeout:
                time.sleep(1.0)
                d = eng.status.decision
                if eng.status.error:
                    raise AssertionError("engine error: %s" % eng.status.error)
                print(
                    "   %4.0fs  in %6.1f dB  %-9s  key %-9s  shift %+d  share %3.0f%%"
                    % (
                        time.perf_counter() - t0,
                        eng.status.input_dbfs,
                        d.status,
                        d.key_name,
                        eng.status.applied_shift,
                        d.leader_share * 100,
                    )
                )
                if d.status == "locked":
                    locked = eng.status.applied_shift
                    break
    finally:
        stop.set()
        t.join(timeout=3)

    ok = locked == want_shift
    print("   -> locked on %s, wanted %+d : %s"
          % ("%+d" % locked if locked is not None else "nothing", want_shift,
             "OK" if ok else "MISS"))
    return ok


if __name__ == "__main__":
    cap = devices.default_capture()
    print("capturing loopback of: %s" % cap.name)
    print("playing through:       %s" % devices.default_output().name)
    results = []
    for pc, minor, label in ((7, False, "G major"), (4, True, "E minor")):
        results.append(run_case(pc, minor, label))
    print("\n%d/%d live cases locked correctly" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)
