"""Routing plan and restore logic, with the Core Audio calls faked.

The real calls change system settings, so these tests substitute a recorder and
assert that whatever was changed is changed back -- including on the crash path,
where only the on-disk undo record can save us.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotranspose import autoroute, devices, winaudio  # noqa: E402

JBL = devices.Device("id-jbl", "Speakers (JBL Charge 5)", 2)
RT = devices.Device("id-rt", "Speakers (2- Realtek(R) Audio)", 2)


class FakeWinAudio:
    def __init__(self, muted=None):
        self.muted = dict(muted or {JBL.id: False, RT.id: False})
        self.default = JBL.id
        self.calls = []

    def endpoints(self):
        return [
            winaudio.Endpoint(id=i, name=n, muted=self.muted.get(i, False), volume=0.5)
            for i, n in ((JBL.id, JBL.name), (RT.id, RT.name))
        ]

    def find(self, device_id):
        return next((e for e in self.endpoints() if e.id == device_id), None)

    def set_default_output(self, device_id):
        self.calls.append(("default", device_id))
        self.default = device_id

    def set_mute(self, device_id, muted):
        self.calls.append(("mute", device_id, muted))
        self.muted[device_id] = muted

    WinAudioUnavailable = winaudio.WinAudioUnavailable


def _isolated(fn):
    """Run a test with the patches in place, then put the real modules back."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        finally:
            restore_patches()

    return wrapper


_ORIG = {
    "winaudio": autoroute.winaudio,
    "list_outputs": devices.list_outputs,
    "default_output": devices.default_output,
    "resolve_output": devices.resolve_output,
    "state_path": autoroute._state_path,
}


def restore_patches():
    """Undo the monkeypatching, so other suites in the same process are unaffected."""
    autoroute.winaudio = _ORIG["winaudio"]
    devices.list_outputs = _ORIG["list_outputs"]
    devices.default_output = _ORIG["default_output"]
    devices.resolve_output = _ORIG["resolve_output"]
    autoroute._state_path = _ORIG["state_path"]


def install(fake, outputs=(JBL, RT), default=JBL, tmp=None):
    autoroute.winaudio = fake
    devices.list_outputs = lambda: list(outputs)
    devices.default_output = lambda: default
    devices.resolve_output = lambda spec: next(
        (d for d in outputs if spec and spec.lower() in d.name.lower()), default
    )
    if tmp is not None:
        autoroute._state_path = lambda: tmp


@_isolated
def test_plan_prefers_your_listening_device():
    fake = FakeWinAudio()
    install(fake)
    plan = autoroute.build_plan()
    print("Plan with JBL as the current default:")
    for line in plan.describe():
        print("  -", line)
    assert plan.sink.id == JBL.id, "should play to the device you already listen on"
    assert plan.source.id == RT.id, "should push apps to the other device"
    assert plan.set_default and plan.mute_source
    print("  OK")


@_isolated
def test_plan_refuses_with_one_device():
    install(FakeWinAudio(), outputs=(JBL,))
    print("\nOne output device only:")
    try:
        autoroute.build_plan()
    except autoroute.RoutingError as exc:
        assert "second output" in str(exc)
        print("  refused with a usable message. OK")
        return
    raise AssertionError("should have refused")


@_isolated
def test_applies_and_restores(tmp_path=Path("./.pytest-state.json")):
    fake = FakeWinAudio()
    install(fake, tmp=tmp_path)
    plan = autoroute.build_plan()
    print("\nApply then restore:")
    with autoroute.Routing(plan):
        assert fake.default == RT.id, "apps were not pointed at the capture device"
        assert fake.muted[RT.id] is True, "capture device was not muted"
        assert tmp_path.exists(), "no crash-recovery record was written"
        print("  during:  default=%s  realtek muted=%s" % (fake.default, fake.muted[RT.id]))
    print("  after:   default=%s  realtek muted=%s" % (fake.default, fake.muted[RT.id]))
    assert fake.default == JBL.id, "default output was not put back"
    assert fake.muted[RT.id] is False, "capture device was left muted"
    assert not tmp_path.exists(), "recovery record was not cleared"
    print("  OK")


@_isolated
def test_does_not_unmute_what_was_already_muted():
    """If the source was muted before we started, leave it muted."""
    fake = FakeWinAudio(muted={JBL.id: False, RT.id: True})
    install(fake, tmp=Path("./.pytest-state2.json"))
    plan = autoroute.build_plan()
    print("\nSource already muted by the user:")
    with autoroute.Routing(plan):
        pass
    assert fake.muted[RT.id] is True, "we unmuted something the user had muted"
    assert ("mute", RT.id, True) not in fake.calls, "muted a device that was already muted"
    print("  left it muted, as the user had it. OK")


@_isolated
def test_crash_recovery(tmp_path=Path("./.pytest-state3.json")):
    """Simulate a kill: apply, drop the object, recover from the on-disk record."""
    fake = FakeWinAudio()
    install(fake, tmp=tmp_path)
    plan = autoroute.build_plan()
    routing = autoroute.Routing(plan)
    routing.__enter__()
    print("\nCrash recovery:")
    print("  mid-run: default=%s  realtek muted=%s" % (fake.default, fake.muted[RT.id]))
    routing._applied = False  # pretend the process died without unwinding
    del routing

    done = autoroute.restore_pending()
    print("  recovery did: %s" % "; ".join(done))
    print("  after:   default=%s  realtek muted=%s" % (fake.default, fake.muted[RT.id]))
    assert fake.default == JBL.id
    assert fake.muted[RT.id] is False
    assert autoroute.pending_restore() is None
    print("  OK")


@_isolated
def test_wont_restore_a_live_instances_settings(tmp_path=Path("./.pytest-state4.json")):
    """A second instance must not undo the routing of one that is still running."""
    fake = FakeWinAudio()
    install(fake, tmp=tmp_path)
    plan = autoroute.build_plan()
    routing = autoroute.Routing(plan)
    routing.__enter__()

    # Rewrite the record as if it belonged to a different, still-live process.
    import json

    data = json.loads(tmp_path.read_text(encoding="utf-8"))
    data["pid"] = 999999
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    real_alive = autoroute._pid_alive
    autoroute._pid_alive = lambda pid: pid == 999999

    print("\nSecond instance, first still running:")
    try:
        assert autoroute.owned_by_live_process() == 999999
        done = autoroute.restore_pending()
        print("  result: %s" % "; ".join(done))
        assert any("skipped" in d for d in done), "should have refused"
        assert fake.default == RT.id, "it undid a live instance's routing"

        forced = autoroute.restore_pending(force=True)
        print("  with --force: %s" % "; ".join(forced))
        assert fake.default == JBL.id, "--force did not restore"
    finally:
        autoroute._pid_alive = real_alive
        routing._applied = False
        tmp_path.unlink(missing_ok=True)
    print("  OK")


def test_dead_pid_is_recoverable(tmp_path=Path("./.pytest-state5.json")):
    """A record from a process that is gone must still be restorable."""
    fake = FakeWinAudio()
    install(fake, tmp=tmp_path)
    plan = autoroute.build_plan()
    routing = autoroute.Routing(plan)
    routing.__enter__()
    import json

    data = json.loads(tmp_path.read_text(encoding="utf-8"))
    data["pid"] = 999998
    tmp_path.write_text(json.dumps(data), encoding="utf-8")
    real_alive = autoroute._pid_alive
    autoroute._pid_alive = lambda pid: False  # that process is gone

    print("\nRecord left by a dead process:")
    try:
        assert autoroute.owned_by_live_process() is None
        done = autoroute.restore_pending()
        print("  recovery did: %s" % "; ".join(done))
        assert fake.default == JBL.id
        assert fake.muted[RT.id] is False
    finally:
        autoroute._pid_alive = real_alive
        routing._applied = False
        tmp_path.unlink(missing_ok=True)
        restore_patches()
    print("  OK")


if __name__ == "__main__":
    for f in (Path("./.pytest-state.json"), Path("./.pytest-state2.json"),
              Path("./.pytest-state3.json"), Path("./.pytest-state4.json")):
        f.unlink(missing_ok=True)
    test_plan_prefers_your_listening_device()
    test_plan_refuses_with_one_device()
    test_applies_and_restores()
    test_does_not_unmute_what_was_already_muted()
    test_crash_recovery()
    test_wont_restore_a_live_instances_settings()
    test_dead_pid_is_recoverable()
    for f in (Path("./.pytest-state.json"), Path("./.pytest-state2.json"),
              Path("./.pytest-state3.json"), Path("./.pytest-state4.json")):
        f.unlink(missing_ok=True)
