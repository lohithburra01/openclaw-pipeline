"""F1 qualifying-replay pipeline.

Produces a 40-second top-3 lap-comparison video for each Qualifying and Sprint
Qualifying session of the season. Drops it in the same Google Drive folder the
race pipeline uses, named ``{location}_{year}_Q.mp4`` or
``{location}_{year}_SQ.mp4`` (the suffix distinguishes Saturday Qualifying from
Friday Sprint Qualifying on sprint weekends).

Mirrors ``race_replay.py`` on the main branch in structure: same OpenF1 source,
same Drive secrets, same self-scheduling cron-rewriting trick, same retry slots.
Visual engine is ported from ``render.py`` (HiFi delta + spline track map +
chasing cars) but the data layer is OpenF1, not FastF1, so it works on a CI
runner. No manifest involvement.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d, splprep, splev
from scipy.signal import correlate, medfilt

# ============================================================
# CONFIGURATION
# ============================================================
SEASON = int(os.environ.get("QUALI_SEASON", datetime.utcnow().year))
QUALI_END_BUFFER_MIN = 75            # Q  ~60 min + cooldown
SPRINT_QUALI_END_BUFFER_MIN = 60     # SQ ~45 min + cooldown
RETRY_OFFSETS_MIN = [15, 60, 105, 150, 195]
LOOKAHEAD_DAYS = 9
RENDER_LOOKBACK_HOURS = 6
WEEKLY_SAFETY_CRON = "0 12 * * 3"     # Wednesday self-heal tick
DURATION_SECONDS = 40
FPS = 30
TOP_N = 3
WIDTH, HEIGHT = 1080, 1920
ZOOM_FACTOR = 3.0
TRAIL_FRAMES = 60
WORKFLOW_PATH = os.path.join(".github", "workflows", "quali_replay.yml")
CRON_BEGIN = "# === BEGIN AUTO-MANAGED CRON (edited by quali_replay.py) ==="
CRON_END = "# === END AUTO-MANAGED CRON ==="
OUTPUT_DIR = "output"
OPENF1_BASE = "https://api.openf1.org/v1"


# ============================================================
# TEAM COLOURS (kept in sync with race_replay.py)
# ============================================================
def get_constructor_color(team_name):
    """Dynamic team name -> strict 2026 hex colour. Anything unknown -> grey."""
    team_name = str(team_name).lower()
    if 'red bull' in team_name and 'racing bulls' not in team_name and 'rb' not in team_name:
        return '#3671C6'
    if 'mclaren' in team_name: return '#FF8000'
    if 'ferrari' in team_name: return '#E80020'
    if 'mercedes' in team_name: return '#27F4D2'
    if 'aston martin' in team_name: return '#2293D1'
    if 'williams' in team_name: return '#66C2FF'
    if 'alpine' in team_name: return '#0090FF'
    if 'haas' in team_name: return '#FFFFFF'
    if 'rb' in team_name or 'racing bulls' in team_name: return '#66CCFF'
    if 'audi' in team_name or 'sauber' in team_name: return '#00E600'
    if 'cadillac' in team_name: return '#FFD700'
    return '#888888'


def hex_to_rgb(hex_col):
    hex_col = hex_col.lstrip('#')
    return tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))


def hex_to_bgr(hex_col):
    r, g, b = hex_to_rgb(hex_col)
    return (b, g, r)


# ============================================================
# OPENF1 HTTP CLIENT (same retry policy as race_replay.py)
# ============================================================
def openf1_get(path, retries=5):
    """GET an OpenF1 endpoint; return the parsed JSON list. Retries on 429/5xx."""
    url = f"{OPENF1_BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "quali-replay"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  OpenF1 returned {exc.code}; retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise


def _iso_to_posix(iso_str):
    return datetime.fromisoformat(iso_str).timestamp()


# ============================================================
# OPENF1 -> TRACE EXTRACTION
# ============================================================
def pick_top_drivers(session_key, top_n=TOP_N):
    """Return the top-N driver_numbers by fastest single lap in the session.

    The three fastest laps in a Qualifying session always come from Q3, and
    likewise for SQ3 in a Sprint Qualifying. So 'fastest lap per driver,
    sort ascending, take top N' gives the Q3/SQ3 shootout participants in
    finishing order, with no need for the (newer, less stable)
    ``session_result`` endpoint.
    """
    laps = openf1_get(f"laps?session_key={session_key}")
    fastest_per_drv = {}
    for l in laps:
        dur = l.get("lap_duration")
        if dur is None:
            continue
        dn = l["driver_number"]
        if dn not in fastest_per_drv or dur < fastest_per_drv[dn]:
            fastest_per_drv[dn] = dur
    ranked = sorted(fastest_per_drv.items(), key=lambda kv: kv[1])
    return [dn for dn, _ in ranked[:top_n]]


def build_driver_trace(session_key, driver_number, driver_info):
    """Build a fastest-lap trace dict for one driver from OpenF1.

    Returns a dict with keys ``dist`` (m), ``speed`` (m/s), ``time`` (s,
    lap-relative), ``x``, ``y``, ``lap_time``, ``driver``, ``team_color``,
    or ``None`` if telemetry is missing/insufficient.
    """
    laps = [
        l for l in openf1_get(
            f"laps?session_key={session_key}&driver_number={driver_number}"
        )
        if l.get("lap_duration") is not None
    ]
    if not laps:
        return None
    fastest = min(laps, key=lambda l: l["lap_duration"])
    lap_start_iso = fastest["date_start"]
    lap_dur = float(fastest["lap_duration"])
    end_dt = datetime.fromisoformat(lap_start_iso) + timedelta(seconds=lap_dur + 1.0)
    end_iso = end_dt.isoformat()

    cd = openf1_get(
        f"car_data?session_key={session_key}&driver_number={driver_number}"
        f"&date>={lap_start_iso}&date<={end_iso}"
    )
    loc = openf1_get(
        f"location?session_key={session_key}&driver_number={driver_number}"
        f"&date>={lap_start_iso}&date<={end_iso}"
    )
    if not cd or not loc:
        return None

    cd.sort(key=lambda r: r["date"])
    loc.sort(key=lambda r: r["date"])

    lap_start = _iso_to_posix(lap_start_iso)
    cd_t = np.array([_iso_to_posix(r["date"]) - lap_start for r in cd], dtype=float)
    cd_v = np.array([(r.get("speed") or 0.0) / 3.6 for r in cd], dtype=float)  # km/h -> m/s
    loc_t = np.array([_iso_to_posix(r["date"]) - lap_start for r in loc], dtype=float)
    loc_x = np.array([(r.get("x") or 0.0) for r in loc], dtype=float) / 10.0
    loc_y = np.array([(r.get("y") or 0.0) for r in loc], dtype=float) / 10.0

    # Keep car_data only within the location time span so interpolation never
    # extrapolates a position from a single endpoint sample.
    valid = (cd_t >= loc_t[0]) & (cd_t <= loc_t[-1])
    cd_t = cd_t[valid]; cd_v = cd_v[valid]
    if len(cd_t) < 10:
        return None

    x = np.interp(cd_t, loc_t, loc_x)
    y = np.interp(cd_t, loc_t, loc_y)

    # OpenF1 occasionally emits same-timestamp duplicates; drop them.
    _, uniq = np.unique(cd_t, return_index=True)
    uniq.sort()
    cd_t = cd_t[uniq]; cd_v = cd_v[uniq]; x = x[uniq]; y = y[uniq]

    dt = np.diff(cd_t, prepend=cd_t[0])
    dist = np.cumsum(cd_v * dt)
    dist = dist - dist[0]

    return {
        "dist": dist,
        "speed": cd_v,
        "time": cd_t,
        "x": x,
        "y": y,
        "lap_time": lap_dur,
        "driver": driver_info.get("name_acronym") or f"#{driver_number}",
        "team_color": get_constructor_color(driver_info.get("team_name", "")),
    }


def load_quali_traces(session_key, top_n=TOP_N):
    """Top-N fastest qualifiers, each as a complete trace, ordered fastest first."""
    drivers = openf1_get(f"drivers?session_key={session_key}")
    drv_by_num = {d["driver_number"]: d for d in drivers}
    top_nums = pick_top_drivers(session_key, top_n=top_n)
    traces = []
    for rank, dn in enumerate(top_nums, start=1):
        tr = build_driver_trace(session_key, dn, drv_by_num.get(dn, {}))
        if tr is None:
            print(f"  WARN: skipping driver {dn} (no usable telemetry)")
            continue
        tr["rank"] = rank
        traces.append(tr)
    return traces


# ============================================================
# HIFI DELTA ENGINE (copied verbatim from render.py)
# ============================================================
def calculate_hifi_delta(ref_trace, tgt_trace):
    """Distance-aligned, residual-corrected time delta between two laps."""
    master_len = ref_trace['dist'].max()
    scale = master_len / tgt_trace['dist'].max()
    tgt_dist_scaled = tgt_trace['dist'] * scale

    grid_len = 10000; window_size = 300; step_size = 50
    common_grid = np.linspace(0, master_len, grid_len)
    v_ref = np.interp(common_grid, ref_trace['dist'], ref_trace['speed'])
    v_tgt = np.interp(common_grid, tgt_dist_scaled, tgt_trace['speed'])

    shifts, positions = [], []
    for start_pos in range(0, int(master_len), step_size):
        end_pos = start_pos + window_size
        if end_pos > master_len:
            break
        i0 = int((start_pos / master_len) * grid_len)
        i1 = int((end_pos / master_len) * grid_len)
        corr = correlate(v_ref[i0:i1], v_tgt[i0:i1], mode='same')
        if len(corr) == 0:
            continue
        lag_idx = np.argmax(corr) - (len(corr) // 2)
        shift_m = lag_idx * (master_len / grid_len)
        if abs(shift_m) < 40:
            shifts.append(shift_m); positions.append(start_pos + window_size / 2)

    if positions:
        if positions[0] > 0:
            positions.insert(0, 0); shifts.insert(0, shifts[0])
        if positions[-1] < master_len:
            positions.append(master_len); shifts.append(shifts[-1])
        positions, shifts = zip(*sorted(zip(positions, shifts)))
        shift_interp = interp1d(positions, shifts, kind='linear', fill_value="extrapolate")
        tgt_dist_warped = tgt_dist_scaled + shift_interp(tgt_dist_scaled)
    else:
        tgt_dist_warped = tgt_dist_scaled

    f_tgt_time = interp1d(tgt_dist_warped, tgt_trace['time'], fill_value="extrapolate")
    raw_delta = f_tgt_time(ref_trace['dist']) - ref_trace['time']
    delta_smooth = medfilt(raw_delta, kernel_size=15)
    delta_zeroed = delta_smooth - delta_smooth[0]

    actual_lap_diff = tgt_trace['lap_time'] - ref_trace['lap_time']
    residual_error = delta_zeroed[-1] - actual_lap_diff
    if abs(residual_error) > 1e-6:
        ramp = np.linspace(0, 1, len(delta_zeroed))
        final_delta = delta_zeroed - (ramp * residual_error)
    else:
        final_delta = delta_zeroed
    return ref_trace['dist'], final_delta


# ============================================================
# VIDEO BAKER
# ============================================================
def _load_fonts():
    """Load the Formula1 font family from CWD; fall back to PIL default."""
    bold = "Formula1-Bold_web_0.ttf.ttf"
    reg = "Formula1-Regular_web_0.ttf.ttf"
    if not (os.path.exists(bold) and os.path.exists(reg)):
        print("  WARN: F1 fonts not found, using default font.")
        d = ImageFont.load_default()
        return {k: d for k in ("title", "sub", "lb_pos", "lb_name", "lb_gap", "car", "wm")}
    return {
        "title":   ImageFont.truetype(bold, 42),
        "sub":     ImageFont.truetype(reg, 26),
        "lb_pos":  ImageFont.truetype(bold, 20),
        "lb_name": ImageFont.truetype(bold, 30),
        "lb_gap":  ImageFont.truetype(reg, 26),
        "car":     ImageFont.truetype(bold, 20),
        "wm":      ImageFont.truetype(bold, 24),
    }


def _draw_centered(draw, x, y, text, font, fill):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    except Exception:
        w, h = len(text) * 8, 14
    draw.text((x - w / 2, y - h / 2 - 4), text, font=font, fill=fill)


def bake_quali_video(traces, location, year, session_label, out_path,
                     duration_seconds=DURATION_SECONDS, fps=FPS,
                     zoom_factor=ZOOM_FACTOR, trail_frames=TRAIL_FRAMES):
    """Render a top-N lap-comparison video to ``out_path``.

    ``traces`` is ordered fastest-first; the first trace is the reference (its
    location samples define the track spline; its lap is the timing baseline).
    The entire lap is compressed into ``duration_seconds`` of wall-clock video
    by sampling sim-time as a fraction of the slowest aligned lap end."""
    if not traces:
        raise RuntimeError("no traces to render")
    ref = traces[0]
    fonts = _load_fonts()

    title1 = f"{location.upper()} {year}"
    title2 = f"TOP {len(traces)} - {session_label.upper()} LAP COMPARISON"

    master_len = ref['dist'][-1]
    time_mappings = {}
    delta_fns = {}

    print("  applying HiFi delta engine...")
    for tr in traces:
        uid = tr['driver']
        if tr is ref:
            mapped = ref['time']
            delta_fns[uid] = lambda _d: 0.0
        else:
            _, fd = calculate_hifi_delta(ref, tr)
            mapped = ref['time'] + fd
            delta_fns[uid] = interp1d(ref['dist'], fd, fill_value="extrapolate")
        time_mappings[uid] = np.maximum.accumulate(mapped)

    # Track spline from the reference driver's centred (x,y).
    ref_x = ref['x'] - np.mean(ref['x'])
    ref_y = ref['y'] - np.mean(ref['y'])
    coords = np.stack([ref_x, ref_y], axis=1)
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mask = np.concatenate([[True], diffs > 0])
    ref_x = ref_x[mask]; ref_y = ref_y[mask]
    if len(ref_x) < 8:
        raise RuntimeError("not enough location samples to build track spline")

    tck, _ = splprep([ref_x, ref_y], s=100, per=0)

    def get_pos(prog):
        p = float(np.clip(prog, 0.0, 1.0))
        x, y = splev(p, tck)
        return float(x), float(y)

    track_pts = [get_pos(p) for p in np.linspace(0, 1, 2000)]
    max_real_time = max(t[-1] for t in time_mappings.values())
    total_frames = int(duration_seconds * fps)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    print(f"  rendering {total_frames} frames @ {fps} fps -> {out_path}")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT)
    )

    track_w = max(p[0] for p in track_pts) - min(p[0] for p in track_pts)
    track_h = max(p[1] for p in track_pts) - min(p[1] for p in track_pts)
    scale = (min(WIDTH, HEIGHT * 0.5) / max(track_w, track_h)) * zoom_factor

    def world_to_screen(x, y, cam_x, cam_y):
        return (int(WIDTH / 2 + (x - cam_x) * scale),
                int(HEIGHT * 0.4 - (y - cam_y) * scale))

    MAP_SIZE = 280
    MAP_X = WIDTH - MAP_SIZE - 60
    lb_y_start = 1350 - (len(traces) * 70)
    MAP_Y = 1350 - MAP_SIZE
    map_scale = MAP_SIZE / max(track_w, track_h) * 0.95

    def world_to_minimap(x, y):
        return (int(MAP_X + MAP_SIZE / 2 + x * map_scale),
                int(MAP_Y + MAP_SIZE / 2 - y * map_scale))

    map_track_screen = np.array(
        [world_to_minimap(p[0], p[1]) for p in track_pts], np.int32
    )
    track_screen_base = np.array(track_pts)

    trails = {tr['driver']: [] for tr in traces}
    car_radius = 30
    ref_uid = ref['driver']
    ref_dist_grid = ref['dist']

    for f in range(total_frames):
        if f % (fps * 5) == 0:
            print(f"    frame {f}/{total_frames}")

        # Compress the full aligned lap into duration_seconds.
        sim_t = (f / max(1, total_frames - 1)) * max_real_time

        current_dist = {}
        for tr in traces:
            uid = tr['driver']
            current_dist[uid] = float(np.interp(sim_t, time_mappings[uid], ref_dist_grid))

        cam_x, cam_y = get_pos(current_dist[ref_uid] / master_len)

        frame_bgr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        pts = np.array(
            [world_to_screen(p[0], p[1], cam_x, cam_y) for p in track_screen_base], np.int32
        )
        cv2.polylines(frame_bgr, [pts.reshape((-1, 1, 2))], False,
                      (255, 255, 255), 40, cv2.LINE_AA)
        cv2.polylines(frame_bgr, [pts.reshape((-1, 1, 2))], False,
                      (40, 40, 40), 32, cv2.LINE_AA)

        draw_order = sorted(traces, key=lambda tr: current_dist[tr['driver']])

        for tr in draw_order:
            uid = tr['driver']
            px, py = get_pos(current_dist[uid] / master_len)
            color_bgr = hex_to_bgr(tr['team_color'])
            trails[uid].append((px, py))
            if len(trails[uid]) > trail_frames:
                trails[uid].pop(0)
            if len(trails[uid]) > 1:
                screen_trail = np.array(
                    [world_to_screen(tx, ty, cam_x, cam_y) for tx, ty in trails[uid]],
                    np.int32,
                )
                cv2.polylines(frame_bgr, [screen_trail.reshape((-1, 1, 2))], False,
                              color_bgr, 18, cv2.LINE_AA)

        cv2.polylines(frame_bgr, [map_track_screen.reshape((-1, 1, 2))], False,
                      (80, 80, 80), 6, cv2.LINE_AA)
        for tr in draw_order:
            uid = tr['driver']
            px, py = get_pos(current_dist[uid] / master_len)
            sx, sy = world_to_screen(px, py, cam_x, cam_y)
            mx, my = world_to_minimap(px, py)
            color_bgr = hex_to_bgr(tr['team_color'])
            cv2.circle(frame_bgr, (sx, sy), car_radius, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (sx, sy), car_radius, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.circle(frame_bgr, (mx, my), 8, color_bgr, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (mx, my), 8, (255, 255, 255), 2, cv2.LINE_AA)

        img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        _draw_centered(draw, WIDTH // 2, 250, title1, fonts["title"], (255, 0, 50))
        _draw_centered(draw, WIDTH // 2, 300, title2, fonts["sub"], (200, 200, 200))

        # Car-tag labels (driver code on a black background, near the car)
        for tr in draw_order:
            uid = tr['driver']
            sx, sy = world_to_screen(*get_pos(current_dist[uid] / master_len), cam_x, cam_y)
            name = tr['driver']
            try:
                bbox = draw.textbbox((0, 0), name, font=fonts["car"])
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = len(name) * 8, 14
            ly = sy - 55
            draw.rectangle(
                [sx - tw / 2 - 8, ly - th / 2 - 6, sx + tw / 2 + 8, ly + th / 2 + 6],
                fill=(0, 0, 0),
            )
            _draw_centered(draw, sx, ly, name, fonts["car"], (255, 255, 255))

        # Leaderboard panel (sorted by who is furthest along the lap)
        sorted_by_pos = sorted(traces, key=lambda tr: current_dist[tr['driver']], reverse=True)
        for i, tr in enumerate(sorted_by_pos):
            uid = tr['driver']
            dist = current_dist[uid]
            color_rgb = hex_to_rgb(tr['team_color'])
            cy = lb_y_start + i * 70 + 15
            dist_to_ref = abs(dist - current_dist[ref_uid])
            if uid == ref_uid:
                gap_text = "POLE"
                color_text_rgb = (150, 150, 150)
            else:
                sign = "+" if dist < current_dist[ref_uid] else "-"
                time_gap = float(delta_fns[uid](dist))
                gap_text = f"{sign}{dist_to_ref:.0f}m ({sign}{abs(time_gap):.3f}s)"
                color_text_rgb = (230, 50, 50) if dist < current_dist[ref_uid] else (50, 230, 50)
            draw.ellipse([80 - 18, cy - 18, 80 + 18, cy + 18], fill=color_rgb)
            _draw_centered(draw, 80, cy, f"P{i + 1}", fonts["lb_pos"], (255, 255, 255))
            draw.text((120, cy - 18), tr['driver'], font=fonts["lb_name"], fill=color_rgb)
            try:
                name_w = draw.textbbox((0, 0), tr['driver'], font=fonts["lb_name"])[2]
            except Exception:
                name_w = len(tr['driver']) * 16
            draw.text((120 + name_w + 35, cy - 14), gap_text, font=fonts["lb_gap"], fill=color_text_rgb)

        _draw_centered(draw, WIDTH // 2, 1450, "@formulytics", fonts["wm"], (150, 150, 150))

        writer.write(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  rendered {out_path}")


# ============================================================
# SCHEDULING & ORCHESTRATION (mirrors race_replay.py / main branch)
# ============================================================
def session_kind_for(name):
    if name == "Qualifying": return "Q"
    if name == "Sprint Qualifying": return "SQ"
    return None


def session_label_for(kind):
    return "SPRINT QUALIFYING" if kind == "SQ" else "QUALIFYING"


def session_end_buffer(kind):
    return SPRINT_QUALI_END_BUFFER_MIN if kind == "SQ" else QUALI_END_BUFFER_MIN


def session_end_time(start, kind):
    return start + timedelta(minutes=session_end_buffer(kind))


def retry_slots(end_time):
    return [end_time + timedelta(minutes=off) for off in RETRY_OFFSETS_MIN]


def cron_for_datetime(dt):
    return f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"


def output_slug(event_name, year, kind):
    """Stable output filename, e.g. 'monaco_2026_Q.mp4', 'miami_2026_SQ.mp4'."""
    base = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    return f"{base}_{year}_{kind}.mp4"


def load_event_schedule(season):
    return openf1_get(f"sessions?year={season}")


def list_sessions(sessions_json):
    """OpenF1 sessions JSON list -> list of Qualifying / Sprint Qualifying dicts."""
    out = []
    for s in sessions_json:
        kind = session_kind_for(s.get("session_name"))
        if kind is None:
            continue
        start = (datetime.fromisoformat(s["date_start"])
                 .astimezone(timezone.utc).replace(tzinfo=None))
        out.append({
            "event": s.get("location") or s.get("circuit_short_name"),
            "kind": kind,
            "start": start,
            "end": session_end_time(start, kind),
            "session_key": int(s["session_key"]),
        })
    return out


def schedulable_sessions(sessions, now):
    last_offset = max(RETRY_OFFSETS_MIN)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)
    out = []
    for s in sessions:
        last_slot = s["end"] + timedelta(minutes=last_offset)
        if last_slot > now and s["start"] < horizon:
            out.append(s)
    return out


def due_sessions(sessions, now):
    out = []
    for s in sessions:
        if s["end"] <= now <= s["end"] + timedelta(hours=RENDER_LOOKBACK_HOURS):
            out.append(s)
    return out


def build_cron_lines(sessions, now):
    lines = {WEEKLY_SAFETY_CRON}
    for s in schedulable_sessions(sessions, now):
        for slot in retry_slots(s["end"]):
            lines.add(cron_for_datetime(slot))
    return sorted(lines)


def rewrite_cron_block(workflow_text, cron_lines):
    if CRON_BEGIN not in workflow_text or CRON_END not in workflow_text:
        raise RuntimeError(
            f"Workflow file is missing the cron markers {CRON_BEGIN!r}/{CRON_END!r}"
        )
    begin = workflow_text.index(CRON_BEGIN)
    end = workflow_text.index(CRON_END)
    if begin >= end:
        raise RuntimeError("Workflow cron markers are out of order")
    head = workflow_text[:begin + len(CRON_BEGIN)]
    tail = workflow_text[end:]
    block = "".join(f'\n    - cron: "{c}"' for c in cron_lines)
    return f"{head}{block}\n    {tail}"


def session_has_data(session_key):
    try:
        return len(openf1_get(f"laps?session_key={session_key}")) > 0
    except Exception as exc:
        print(f"  OpenF1 laps not ready for session {session_key}: {exc}")
        return False


def openf1_find_session(year, gp, kind):
    """Resolve a (year, gp-name, Q/SQ) to (session_key, canonical location)."""
    target = "Sprint Qualifying" if kind == "SQ" else "Qualifying"
    gp_l = gp.lower()
    for s in openf1_get(f"sessions?year={year}"):
        if s.get("session_name") != target:
            continue
        hay = " ".join(
            str(s.get(k, "")).lower()
            for k in ("location", "circuit_short_name", "country_name")
        )
        if any(w in hay for w in gp_l.split()):
            return int(s["session_key"]), s.get("location") or s.get("circuit_short_name")
    raise RuntimeError(f"OpenF1: no {target} session found for {year} {gp}")


# ============================================================
# GOOGLE DRIVE (same OAuth pattern as race_replay.py)
# ============================================================
_GOOGLE_CREDS = None


def get_google_credentials():
    global _GOOGLE_CREDS
    if _GOOGLE_CREDS is not None:
        return _GOOGLE_CREDS
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(Request())
    _GOOGLE_CREDS = creds
    return creds


def get_drive_service():
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=get_google_credentials())


def drive_has_file(service, folder_id, filename):
    safe = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    return bool(resp.get("files"))


def drive_upload_file(service, folder_id, path):
    from googleapiclient.http import MediaFileUpload
    metadata = {"name": os.path.basename(path), "parents": [folder_id]}
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    result = service.files().create(
        body=metadata, media_body=media, fields="id,name"
    ).execute()
    return result["id"]


def commit_and_push(path, message):
    """Commit a single file and push. Only acts inside GitHub Actions."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        print(f"  (local run) would commit and push {path}")
        return
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email",
         "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "add", path], check=True)
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "pull", "--rebase"], check=True)
    subprocess.run(["git", "push"], check=True)
    print(f"  committed and pushed {path}")


# ============================================================
# CLI ARGS
# ============================================================
def parse_force_session(text):
    """'2024 Monaco Q' -> (year, event, kind). Event name may contain spaces."""
    parts = text.split()
    if len(parts) < 3:
        raise SystemExit(
            '--force-session must be "YEAR EVENT KIND", e.g. "2024 Monaco Q"'
        )
    year = int(parts[0])
    kind = parts[-1].upper()
    if kind not in ("Q", "SQ"):
        raise SystemExit(f'--force-session kind must be Q or SQ, got "{kind}"')
    event = " ".join(parts[1:-1])
    return year, event, kind


# ============================================================
# PHASES
# ============================================================
def reschedule(now, dry_run=False):
    """Phase 1: rewrite the workflow's cron block from the F1 calendar."""
    sessions = list_sessions(load_event_schedule(SEASON))
    cron_lines = build_cron_lines(sessions, now)
    print("Cron block to apply:")
    for c in cron_lines:
        print(f'  - cron: "{c}"')
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        current = f.read()
    updated = rewrite_cron_block(current, cron_lines)
    if updated == current:
        print("Schedule unchanged.")
        return
    if dry_run:
        print("(dry-run) workflow file not modified.")
        return
    with open(WORKFLOW_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    commit_and_push(WORKFLOW_PATH, "chore: update quali-replay cron schedule")
    print("Schedule updated.")


def render_one(session_key, location, year, kind, out_path):
    print(f"  fetching top-{TOP_N} traces from OpenF1...")
    traces = load_quali_traces(session_key, top_n=TOP_N)
    if len(traces) < 2:
        raise RuntimeError(
            f"only {len(traces)} traces available — need at least 2 for a comparison"
        )
    bake_quali_video(traces, location, year, session_label_for(kind), out_path)


def render_due(now):
    """Phase 2: render any just-ended Quali/SQ session not already in Drive."""
    sessions = list_sessions(load_event_schedule(SEASON))
    candidates = due_sessions(sessions, now)
    if not candidates:
        print("NO_SESSION")
        return
    service = get_drive_service()
    folder = os.environ["GDRIVE_FOLDER_ID"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for s in candidates:
        slug = output_slug(s["event"], SEASON, s["kind"])
        print(f"Candidate: {s['event']} ({s['kind']}) -> {slug}")
        if drive_has_file(service, folder, slug):
            print(f"  ALREADY_RENDERED ({slug})")
            continue
        if not session_has_data(s["session_key"]):
            print(f"  DATA_NOT_READY ({s['event']} {s['kind']})")
            continue
        out_path = os.path.join(OUTPUT_DIR, slug)
        try:
            render_one(s["session_key"], s["event"], SEASON, s["kind"], out_path)
        except Exception as exc:
            print(f"  RENDER_ERROR ({slug}): {exc}")
            continue
        file_id = drive_upload_file(service, folder, out_path)
        print(f"  UPLOADED ({slug}, drive id {file_id})")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="F1 qualifying-replay pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cron block without writing or committing")
    parser.add_argument("--force-session", metavar="SPEC",
                        help='render one session locally, e.g. "2024 Monaco Q"')
    parser.add_argument("--test-session", metavar="SPEC", default="",
                        help='end-to-end test: render a past session AND upload '
                             'it to Google Drive as TEST_<name>, e.g. "2024 Monaco Q"')
    args = parser.parse_args()

    if args.force_session:
        year, event, kind = parse_force_session(args.force_session)
        sk, loc = openf1_find_session(year, event, kind)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, output_slug(loc, year, kind))
        render_one(sk, loc, year, kind, out_path)
        print(f"Rendered {out_path}")
        return

    if args.test_session:
        year, event, kind = parse_force_session(args.test_session)
        sk, loc = openf1_find_session(year, event, kind)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "TEST_" + output_slug(loc, year, kind))
        print(f"=== END-TO-END TEST: {loc} {kind} ({year}) ===")
        render_one(sk, loc, year, kind, out_path)
        try:
            service = get_drive_service()
            folder = os.environ["GDRIVE_FOLDER_ID"]
            file_id = drive_upload_file(service, folder, out_path)
            print(f"UPLOADED test video to Google Drive (file id {file_id})")
        except Exception as exc:
            print(f"NOTE: Drive upload skipped ({type(exc).__name__}: {exc})")
            print("The rendered video is still saved as a workflow artifact.")
        return

    now = datetime.utcnow()
    print(f"=== quali_replay.py @ {now} UTC (season {SEASON}) ===")
    reschedule(now, dry_run=args.dry_run)
    if not args.dry_run:
        render_due(now)


if __name__ == "__main__":
    main()
