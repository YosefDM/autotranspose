"""Windows Core Audio operations, run as a subprocess.

This lives in its own process on purpose. `comtypes` initialises COM as a
single-threaded apartment while `soundcard` wants a multi-threaded one, and
whichever loads second fails with "Cannot change thread mode after it is set".
Keeping every COM call over here means the main process never imports comtypes,
so the two never meet.

Usage (JSON on stdout):
    python -m autotranspose._winaudio_helper list
    python -m autotranspose._winaudio_helper set-default <endpoint-id>
    python -m autotranspose._winaudio_helper set-mute <endpoint-id> <0|1>
"""
from __future__ import annotations

import json
import sys
from ctypes import HRESULT, POINTER, c_int, cast
from ctypes.wintypes import LPCWSTR

from comtypes import COMMETHOD, GUID, CLSCTX_ALL, CoCreateInstance, IUnknown
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Undocumented, but the only way to set the default endpoint, and what every
# tool that does this (nircmd, SoundVolumeView, AudioDeviceCmdlets) relies on.
CLSID_POLICY_CONFIG_CLIENT = GUID("{870af99c-171d-4f9e-af0d-e63df40c2bc9}")
ROLES = (0, 1, 2)  # eConsole, eMultimedia, eCommunications


class IPolicyConfig(IUnknown):
    _iid_ = GUID("{f8679f50-850a-41cf-9c72-430f290290c8}")
    _methods_ = (
        COMMETHOD([], HRESULT, "GetMixFormat"),
        COMMETHOD([], HRESULT, "GetDeviceFormat"),
        COMMETHOD([], HRESULT, "ResetDeviceFormat"),
        COMMETHOD([], HRESULT, "SetDeviceFormat"),
        COMMETHOD([], HRESULT, "GetProcessingPeriod"),
        COMMETHOD([], HRESULT, "SetProcessingPeriod"),
        COMMETHOD([], HRESULT, "GetShareMode"),
        COMMETHOD([], HRESULT, "SetShareMode"),
        COMMETHOD([], HRESULT, "GetPropertyValue"),
        COMMETHOD([], HRESULT, "SetPropertyValue"),
        COMMETHOD(
            [], HRESULT, "SetDefaultEndpoint",
            (["in"], LPCWSTR, "deviceId"), (["in"], c_int, "role"),
        ),
        COMMETHOD([], HRESULT, "SetEndpointVisibility"),
    )


def _endpoint_volume(device):
    iface = device._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))


def _render_devices():
    """Active render endpoints only; Windows keeps stale duplicates around."""
    out = []
    for d in AudioUtilities.GetAllDevices():
        if not d.id or "Active" not in str(getattr(d, "state", "")):
            continue
        entry = {"id": d.id, "name": d.FriendlyName}
        try:
            vol = _endpoint_volume(d)
            entry["muted"] = bool(vol.GetMute())
            entry["volume"] = round(float(vol.GetMasterVolumeLevelScalar()), 4)
        except Exception:
            # Capture endpoints and anything that refuses to activate.
            continue
        out.append(entry)
    return out


def _find(device_id):
    for d in AudioUtilities.GetAllDevices():
        if d.id == device_id:
            return d
    raise SystemExit(json.dumps({"ok": False, "error": f"no endpoint {device_id!r}"}))


def main(argv):
    if not argv:
        print(json.dumps({"ok": False, "error": "no command"}))
        return 2
    cmd = argv[0]

    if cmd == "list":
        print(json.dumps({"ok": True, "devices": _render_devices()}))
        return 0

    if cmd == "set-default":
        pc = CoCreateInstance(CLSID_POLICY_CONFIG_CLIENT, IPolicyConfig, CLSCTX_ALL)
        for role in ROLES:
            pc.SetDefaultEndpoint(argv[1], role)
        print(json.dumps({"ok": True}))
        return 0

    if cmd == "set-mute":
        vol = _endpoint_volume(_find(argv[1]))
        vol.SetMute(int(argv[2]), None)
        print(json.dumps({"ok": True, "muted": bool(vol.GetMute())}))
        return 0

    print(json.dumps({"ok": False, "error": f"unknown command {cmd!r}"}))
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:  # always answer in JSON so the caller can parse it
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
