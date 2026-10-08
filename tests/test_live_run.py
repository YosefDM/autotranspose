"""The real thing, end to end, on two real devices.

Plays a song in a known key out of one output device, runs the full app
(capture -> detect -> shift -> play) into a second output device, then captures
*that* device's loopback and runs key detection on what actually came out. The
assertion is the one that matters: the audio reaching your ears is in C.

Audible on both devices. Needs two output devices; pass their name fragments:

    python tests/test_live_run.py --source "Realtek" --sink "JBL"
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundcard as sc  # noqa: E402
import soxr  # noqa: E402
from synth import synth_song  # noqa: E402

from autotranspose import devices  # noqa: E402
from autotranspose.engine import Engine, EngineConfig  # noqa: E402
from autotranspose.keydetect import ANALYSIS_SR, detect_key_offline  # noqa: E402
from autotranspose.keyprofiles import shift_class_for_key, signed_shift  # noqa: E402

SR = 48000


def play_loop(device, song, stop, volume):
    spk = sc.get_speaker(device.id)
    stereo = (np.stack([song, song], axis=1) * volume).astype(np.float32)
    with spk.player(samplerate=SR, channels=2, blocksize=2048) as p:
        while not stop.is_set():
            p.play(stereo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Realtek", help="device the music plays out of")
    ap.add_argument("--sink", default="JBL", help="device the transposed audio goes to")
    ap.add_argument("--tonic", type=int, default=4, help="tonic pitch class of the test song")
    ap.add_argument("--minor", action="store_true", default=True)
    ap.add_argument("--volume", type=float, default=0.30)
    ap.add_argument("--timeout", type=float, default=50.0)
    args = ap.parse_args()

    source = devices.resolve_output(args.source)
    sink = devices.resolve_output(args.sink)
    capture = devices.resolve_capture(args.source)  # loopback of the source
    verify = devices.resolve_capture(args.sink)  # loopback of the sink

    want_shift = signed_shift(shift_class_for_key(args.tonic + (12 if args.minor else 0)))
    song = synth_song(args.tonic, minor_key=args.minor, duration=20.0, sr=SR, seed=21)

    print("music out of:    %s" % source.name)
    print("app captures:    %s" % capture.name)
    print("app plays to:    %s" % sink.name)
    print("verifying via:   %s (loopback)" % verify.name)
    print("test song wants: %+d semitones\n" % want_shift)

    chk = devices.check_routing(capture, sink)
    print("routing check:   %s\n" % chk.message)
    if not chk.ok:
        print(chk.advice)
        return 2

    stop = threading.Event()
    player = threading.Thread(target=play_loop, args=(source, song, stop, args.volume), daemon=True)
    player.start()
    time.sleep(1.5)

    cfg = EngineConfig(samplerate=SR, blocksize=1024, min_heard_seconds=10.0)
    eng = Engine(capture, sink, cfg)
    captured = {}

    def record_output(seconds=14.0):
        mic = sc.get_microphone(verify.id, include_loopback=True)
        with mic.recorder(samplerate=SR, channels=2, blocksize=1024) as r:
            captured["data"] = r.record(numframes=int(SR * seconds))

    locked_shift = None
    try:
        with eng:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.timeout:
                time.sleep(1.0)
                d = eng.status.decision
                if eng.status.error:
                    print("ENGINE ERROR: %s" % eng.status.error)
                    return 1
                print(
                    "  %4.0fs  %-9s  in %6.1f dB  out %6.1f dB  key %-9s  shift %+d  "
                    "load %4.1f%%  gaps %d"
                    % (
                        time.perf_counter() - t0,
                        d.status,
                        eng.status.input_dbfs,
                        eng.status.output_dbfs,
                        d.key_name,
                        eng.status.applied_shift,
                        eng.status.audio_load * 100,
                        eng.status.discontinuities,
                    )
                )
                if d.status == "locked":
                    locked_shift = eng.status.applied_shift
                    break

            if locked_shift is None:
                print("\nnever locked within %.0fs" % args.timeout)
                return 1

            print("\nlocked on %+d. Recording what the app is actually outputting..."
                  % locked_shift)
            record_output()
    finally:
        stop.set()
        player.join(timeout=3)

    out = captured.get("data")
    if out is None or np.abs(out).max() < 1e-4:
        print("captured nothing from %s -- is it actually receiving audio?" % sink.name)
        return 1

    mono = out.mean(axis=1).astype(np.float32)
    print("captured %.1fs from the sink, peak %.3f" % (len(mono) / SR, np.abs(mono).max()))

    est = detect_key_offline(soxr.resample(mono, SR, ANALYSIS_SR), ANALYSIS_SR)
    print("\nkey of the audio reaching your ears: %s (shift class %d, conf %.1f%%)"
          % (est.key_name, est.shift_class, est.key_confidence * 100))

    shift_ok = locked_shift == want_shift
    landed = est.shift_class == 0
    print("\n  applied shift      %+d, wanted %+d        : %s"
          % (locked_shift, want_shift, "OK" if shift_ok else "MISS"))
    print("  output on white keys (class 0)       : %s"
          % ("OK" if landed else "MISS (class %d)" % est.shift_class))
    ok = shift_ok and landed
    print("\n%s" % ("PASS - the full run path works on real hardware." if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
