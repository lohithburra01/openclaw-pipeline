"""Direct client for F1's official livetiming static feed.

No third-party wrappers (no FastF1, no OpenF1). Only the Python stdlib —
``urllib`` + ``json`` + ``zlib`` + ``base64``. Designed to give the
race/quali renderers exactly the data shapes they need (per-driver lap
history for race, per-lap telemetry trace for quali).

Endpoint root: ``https://livetiming.formula1.com/static/``.

File formats we handle:
- ``Index.json`` (and other ``.json``): plain JSON, sometimes UTF-8 BOM
  prefixed.
- ``*.jsonStream``: newline-delimited records, each line of the form
  ``"HH:MM:SS.sss<payload>"`` where the payload is plain JSON.
- ``*.z.jsonStream``: same record framing, but the payload is a
  JSON-string-quoted, base64-encoded, zlib-compressed (raw deflate, no
  header) blob of JSON.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone

LIVETIMING_BASE = "https://livetiming.formula1.com/static"
UA = "Mozilla/5.0 (compatible; openclaw-pipeline)"


# ============================================================
# HTTP
# ============================================================
def _fetch(url, retries=5, timeout=60):
    """HTTP GET with retry on 429/5xx. Returns raw response bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  livetiming {exc.code}; retry in {wait}s")
                time.sleep(wait)
                continue
            raise


def _strip_bom(b):
    return b[3:] if b[:3] == b"\xef\xbb\xbf" else b


def _load_json_bytes(b):
    return json.loads(_strip_bom(b).decode("utf-8", errors="replace"))


# ============================================================
# STREAM PARSING
# ============================================================
_TS_RE = re.compile(rb"^(\d\d:\d\d:\d\d(?:\.\d{1,3})?)")


def _split_line(line):
    """Split a stream line into (timestamp_seconds, payload_bytes). Returns
    None for unparseable lines (e.g. junk at end of file)."""
    m = _TS_RE.match(line)
    if not m:
        return None
    ts = m.group(1).decode()
    rest = line[len(ts):]
    if rest[:1] == b" ":
        rest = rest[1:]
    h, mi, se = ts.split(":")
    secs = int(h) * 3600 + int(mi) * 60 + float(se)
    return secs, rest


def parse_jsonstream(b):
    """``*.jsonStream`` -> list of ``(ts_seconds, json_value)``."""
    out = []
    for line in b.splitlines():
        if not line:
            continue
        split = _split_line(line)
        if split is None:
            continue
        ts, payload = split
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        out.append((ts, data))
    return out


def _zlib_decompress_any(raw):
    """Try raw deflate first (F1's actual format), then zlib-with-header."""
    try:
        return zlib.decompress(raw, -zlib.MAX_WBITS)
    except zlib.error:
        return zlib.decompress(raw)


def parse_z_jsonstream(b):
    """``*.z.jsonStream`` -> list of ``(ts_seconds, json_value)``.

    Per line: timestamp + JSON-string-quoted, base64-encoded, raw-deflate-
    compressed JSON body.
    """
    out = []
    for line in b.splitlines():
        if not line:
            continue
        split = _split_line(line)
        if split is None:
            continue
        ts, payload = split
        try:
            b64 = json.loads(payload)  # strips the surrounding JSON quotes
        except Exception:
            continue
        if not isinstance(b64, str):
            continue
        try:
            raw = base64.b64decode(b64)
            decompressed = _zlib_decompress_any(raw)
        except Exception:
            continue
        try:
            data = json.loads(decompressed.decode("utf-8", errors="replace"))
        except Exception:
            continue
        out.append((ts, data))
    return out


# ============================================================
# HIGH-LEVEL FETCHERS
# ============================================================
def get_season_index(year):
    """Return the parsed Index.json for ``year``: meetings + sessions + paths."""
    return _load_json_bytes(_fetch(f"{LIVETIMING_BASE}/{year}/Index.json"))


def get_session_index(session_path):
    """List of feeds available for a session folder.

    ``session_path`` is the ``Path`` field from the season Index, e.g.
    ``'2024/2024-05-26_Monaco_Grand_Prix/2024-05-25_Qualifying/'``.
    """
    return _load_json_bytes(_fetch(f"{LIVETIMING_BASE}/{session_path}Index.json"))


def get_driver_list(session_path):
    """Parsed DriverList.json for the session."""
    return _load_json_bytes(_fetch(f"{LIVETIMING_BASE}/{session_path}DriverList.json"))


def get_session_info(session_path):
    """Parsed SessionInfo.json for the session (UTC start time, etc.)."""
    return _load_json_bytes(_fetch(f"{LIVETIMING_BASE}/{session_path}SessionInfo.json"))


def get_timing_data_stream(session_path):
    """List of ``(ts_seconds, payload)`` from TimingData.jsonStream."""
    raw = _fetch(f"{LIVETIMING_BASE}/{session_path}TimingData.jsonStream")
    return parse_jsonstream(raw)


def get_timing_app_data_stream(session_path):
    """List of ``(ts_seconds, payload)`` from TimingAppData.jsonStream (stints)."""
    raw = _fetch(f"{LIVETIMING_BASE}/{session_path}TimingAppData.jsonStream")
    return parse_jsonstream(raw)


def get_car_data_stream(session_path):
    """Decompressed CarData.z.jsonStream — telemetry channels per driver."""
    raw = _fetch(f"{LIVETIMING_BASE}/{session_path}CarData.z.jsonStream")
    return parse_z_jsonstream(raw)


def get_position_stream(session_path):
    """Decompressed Position.z.jsonStream — track x/y/z per driver."""
    raw = _fetch(f"{LIVETIMING_BASE}/{session_path}Position.z.jsonStream")
    return parse_z_jsonstream(raw)


# ============================================================
# SESSION LOOKUP
# ============================================================
# F1's session Type/Name vocabulary -> our (R, S, Q, SQ) kind codes.
_SESSION_NAME_TO_KIND = {
    "Race": "R",
    "Sprint": "S",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",  # 2023 sprint format
}


def list_season_sessions(year):
    """Flatten the season index into a list of session dicts.

    Each dict has: ``event`` (Location string), ``meeting_name``, ``kind``
    (R / S / Q / SQ), ``start_utc`` (datetime, naive UTC), ``path`` (the
    static-folder path used by the fetchers above), plus the raw ``Name``
    and ``Type`` for debugging.
    """
    idx = get_season_index(year)
    out = []
    for m in idx.get("Meetings", []):
        meeting_name = m.get("Name", "")
        location = m.get("Location") or m.get("Country", {}).get("Name", "")
        for s in m.get("Sessions", []):
            name = s.get("Name", "")
            kind = _SESSION_NAME_TO_KIND.get(name)
            if kind is None:
                continue
            start_str = s.get("StartDate")
            gmt_offset = s.get("GmtOffset", "00:00:00")
            if not start_str:
                continue
            # StartDate is local; convert to UTC via GmtOffset (e.g. "02:00:00").
            try:
                local = datetime.fromisoformat(start_str)
                sign = 1
                gmt = gmt_offset
                if gmt.startswith("-"):
                    sign = -1
                    gmt = gmt[1:]
                gh, gm, gs = (int(x) for x in gmt.split(":"))
                offset_seconds = sign * (gh * 3600 + gm * 60 + gs)
                start_utc = local - _td(offset_seconds)
            except Exception:
                continue
            out.append({
                "event": location,
                "meeting_name": meeting_name,
                "kind": kind,
                "start_utc": start_utc,
                "path": s.get("Path", ""),
                "raw_name": name,
                "raw_type": s.get("Type", ""),
            })
    return out


def _td(seconds):
    from datetime import timedelta
    return timedelta(seconds=seconds)


def find_session(year, event, kind):
    """Resolve (year, event-substring-case-insensitive, kind) -> a session
    dict from ``list_season_sessions``. Raises if not found."""
    needle = (event or "").lower()
    target_kind = kind.upper()
    for s in list_season_sessions(year):
        if s["kind"] != target_kind:
            continue
        hay = " ".join([
            (s.get("event") or "").lower(),
            (s.get("meeting_name") or "").lower(),
        ])
        if any(w in hay for w in needle.split() if w):
            return s
    raise LookupError(f"No {target_kind} session for {year} {event!r}")


# ============================================================
# TIME ALIGNMENT + TRACE EXTRACTION
# ============================================================
def _parse_utc(s):
    """ISO8601 with optional fractional seconds and trailing Z -> naive UTC dt."""
    from datetime import datetime
    s = s.rstrip("Z")
    if "." in s:
        head, frac = s.split(".", 1)
        # Some F1 timestamps carry sub-microsecond precision; truncate.
        frac = (frac + "000000")[:6]
        s = head + "." + frac
    return datetime.fromisoformat(s)


def broadcast_utc_start(car_data_stream):
    """Derive the UTC moment that corresponds to stream-time 0.

    First CarData record has both a stream timestamp (when it was published)
    and an internal Entries[0].Utc (when the sample was taken). The offset
    between them is the broadcast start in UTC."""
    from datetime import timedelta
    if not car_data_stream:
        raise RuntimeError("car_data_stream is empty — cannot anchor UTC")
    first_ts, payload = car_data_stream[0]
    entries = payload.get("Entries") or []
    if not entries:
        raise RuntimeError("first CarData record has no entries")
    first_utc = _parse_utc(entries[0]["Utc"])
    return first_utc - timedelta(seconds=first_ts)


def driver_fastest_lap(timing_data_stream, driver_number):
    """Find a driver's fastest lap from the TimingData stream.

    Returns ``(lap_number, lap_seconds, end_stream_ts)`` or ``None`` if the
    driver has no recorded laps. ``end_stream_ts`` is the stream-relative
    timestamp at which the LastLapTime update arrived — i.e. ~the moment
    the lap ended."""
    dn = str(driver_number)
    fastest_secs = None
    fastest_end_ts = None
    fastest_lap = None
    running_lap_count = None
    for ts, rec in timing_data_stream:
        lines = rec.get("Lines")
        if not isinstance(lines, dict):
            continue
        line = lines.get(dn)
        if not isinstance(line, dict):
            continue
        n = line.get("NumberOfLaps")
        if isinstance(n, int):
            running_lap_count = n
        last = line.get("LastLapTime")
        if isinstance(last, dict):
            v = last.get("Value")
            if v and ":" in v:
                try:
                    m, rest = v.split(":")
                    secs = int(m) * 60 + float(rest)
                except ValueError:
                    continue
                if fastest_secs is None or secs < fastest_secs:
                    fastest_secs = secs
                    fastest_end_ts = ts
                    fastest_lap = running_lap_count
    if fastest_secs is None:
        return None
    return fastest_lap, fastest_secs, fastest_end_ts


def lap_trace_for_driver(car_data_stream, position_stream,
                         broadcast_start, lap_start_utc, lap_end_utc,
                         driver_number):
    """Slice CarData + Position to one lap window and return the rendererr's
    trace dict, or ``None`` if there aren't enough samples.

    Output keys: ``dist`` (m), ``speed`` (m/s), ``time`` (s, lap-relative),
    ``x``, ``y``, ``lap_time``."""
    import numpy as np
    dn = str(driver_number)

    cd_pairs = []  # (utc, speed_kmh)
    for _, payload in car_data_stream:
        for ent in payload.get("Entries", []):
            e_utc = _parse_utc(ent["Utc"])
            if lap_start_utc <= e_utc <= lap_end_utc:
                car = ent.get("Cars", {}).get(dn)
                if car:
                    spd = (car.get("Channels") or {}).get("2")
                    if spd is not None:
                        cd_pairs.append((e_utc, spd))

    pos_pairs = []  # (utc, x, y)
    for _, payload in position_stream:
        for ent in payload.get("Position", []):
            p_utc = _parse_utc(ent["Timestamp"])
            if lap_start_utc <= p_utc <= lap_end_utc:
                ee = (ent.get("Entries") or {}).get(dn)
                if ee:
                    pos_pairs.append((p_utc, ee.get("X", 0), ee.get("Y", 0)))

    if len(cd_pairs) < 10 or len(pos_pairs) < 10:
        return None

    t0 = lap_start_utc
    cd_t = np.array([(u - t0).total_seconds() for u, _ in cd_pairs], dtype=float)
    cd_v = np.array([s / 3.6 for _, s in cd_pairs], dtype=float)  # km/h -> m/s
    pos_t = np.array([(u - t0).total_seconds() for u, _, _ in pos_pairs], dtype=float)
    pos_x = np.array([x / 10.0 for _, x, _ in pos_pairs], dtype=float)
    pos_y = np.array([y / 10.0 for _, _, y in pos_pairs], dtype=float)

    # Keep car samples only within the location time span.
    mask = (cd_t >= pos_t[0]) & (cd_t <= pos_t[-1])
    cd_t = cd_t[mask]; cd_v = cd_v[mask]
    if len(cd_t) < 10:
        return None

    x = np.interp(cd_t, pos_t, pos_x)
    y = np.interp(cd_t, pos_t, pos_y)

    # Drop duplicate timestamps (rare but happens at chunk boundaries).
    _, uniq = np.unique(cd_t, return_index=True)
    uniq.sort()
    cd_t = cd_t[uniq]; cd_v = cd_v[uniq]; x = x[uniq]; y = y[uniq]

    dt = np.diff(cd_t, prepend=cd_t[0])
    dist = np.cumsum(cd_v * dt)
    dist = dist - dist[0]
    lap_time = (lap_end_utc - lap_start_utc).total_seconds()

    return {
        "dist": dist,
        "speed": cd_v,
        "time": cd_t,
        "x": x,
        "y": y,
        "lap_time": lap_time,
    }


def load_quali_traces_raw(year, event, kind, top_n=3):
    """Top-N qualifiers' fastest-lap traces, ready to feed the renderer.

    Returns a list of trace dicts (see ``lap_trace_for_driver``) extended
    with ``driver`` (TLA) and ``team_color`` (hex). Ordered fastest first.
    Uses only the raw livetiming static feed — no FastF1, no OpenF1.
    """
    from datetime import timedelta

    session = find_session(year, event, kind)
    print(f"  resolved: {session['path']}")

    drivers = get_driver_list(session["path"])
    timing = get_timing_data_stream(session["path"])
    car_data = get_car_data_stream(session["path"])
    position = get_position_stream(session["path"])
    print(f"  drivers={len(drivers)}  timing={len(timing)}  car_data={len(car_data)}  position={len(position)}")

    anchor = broadcast_utc_start(car_data)
    print(f"  broadcast_utc_start = {anchor}")

    # Rank drivers by their fastest lap across the whole quali session
    ranked = []
    for dn in drivers.keys():
        res = driver_fastest_lap(timing, dn)
        if res is None:
            continue
        _, secs, _ = res
        ranked.append((dn, secs, res))
    ranked.sort(key=lambda x: x[1])
    ranked = ranked[:top_n]

    traces = []
    for rank, (dn, secs, (lap_num, lap_secs, end_ts)) in enumerate(ranked, start=1):
        lap_end_utc = anchor + timedelta(seconds=end_ts)
        lap_start_utc = lap_end_utc - timedelta(seconds=lap_secs)
        tr = lap_trace_for_driver(car_data, position, anchor,
                                  lap_start_utc, lap_end_utc, dn)
        if tr is None:
            print(f"  WARN: trace for #{dn} too sparse, skipped")
            continue
        info = drivers.get(dn, {}) or {}
        tla = info.get("Tla") or f"#{dn}"
        team = (info.get("TeamName") or "").lower()
        tr["driver"] = tla
        tr["rank"] = rank
        tr["team_color"] = _quick_team_color(team)
        traces.append(tr)
    return traces


def _quick_team_color(team_name):
    """Inline 2026 team-colour table for quick use in this module."""
    t = (team_name or "").lower()
    if "red bull" in t and "racing bulls" not in t and "rb" not in t: return "#3671C6"
    if "mclaren" in t: return "#FF8000"
    if "ferrari" in t: return "#E80020"
    if "mercedes" in t: return "#27F4D2"
    if "aston martin" in t: return "#2293D1"
    if "williams" in t: return "#66C2FF"
    if "alpine" in t: return "#0090FF"
    if "haas" in t: return "#FFFFFF"
    if "rb" in t or "racing bulls" in t: return "#66CCFF"
    if "audi" in t or "sauber" in t: return "#00E600"
    if "cadillac" in t: return "#FFD700"
    return "#888888"
