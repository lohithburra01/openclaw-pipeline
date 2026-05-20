import fastf1
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# ============================================================
# RACE REPLAY PIPELINE - configuration
# ============================================================
import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta

SEASON = int(os.environ.get("RACE_SEASON", datetime.utcnow().year))
RACE_END_BUFFER_MIN = 135       # race counted as "ended" 2h15m after start
SPRINT_END_BUFFER_MIN = 75      # sprint counted as "ended" 1h15m after start
RETRY_OFFSETS_MIN = [15, 60, 105, 150, 195]  # retry slots (minutes after end)
LOOKAHEAD_DAYS = 9              # how far ahead to write cron slots
RENDER_LOOKBACK_HOURS = 6       # window after end-time in which a session renders
WEEKLY_SAFETY_CRON = "0 12 * * 3"            # Wednesday self-heal tick
DURATION_SECONDS = 30
FPS = 60
PORTRAIT = True
WORKFLOW_PATH = os.path.join(".github", "workflows", "race_replay.yml")
CRON_BEGIN = "# === BEGIN AUTO-MANAGED CRON (edited by race_replay.py) ==="
CRON_END = "# === END AUTO-MANAGED CRON ==="
CACHE_DIR = ".fastf1_cache"
OUTPUT_DIR = "output"

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# --- CORE CONFIGURATION ---
GLOBAL_SCALE = 0.90         
VISUAL_GAP_FACTOR = 2.0     
# --------------------------

def get_constructor_color(team_name):
    """Maps the dynamic team name from the session to a strict 2026 color library."""
    team_name = str(team_name).lower()
    
    # 2026 Constructor Color Library
    if 'red bull' in team_name and 'racing bulls' not in team_name and 'rb' not in team_name: 
        return '#3671C6' # Oracle Red Bull Racing
    if 'mclaren' in team_name: return '#FF8000'
    if 'ferrari' in team_name: return '#E80020'
    if 'mercedes' in team_name: return '#27F4D2'
    if 'aston martin' in team_name: return '#2293D1'
    if 'williams' in team_name: return '#66C2FF'
    if 'alpine' in team_name: return '#0090FF'
    if 'haas' in team_name: return '#FFFFFF'
    if 'rb' in team_name or 'racing bulls' in team_name: return '#66CCFF' # Racing Bulls
    if 'audi' in team_name or 'sauber' in team_name: return '#00E600' # Audi Revolut
    if 'cadillac' in team_name: return '#FFD700' # Cadillac Formula 1 Team
    
    return '#888888'

def get_tire_info(compound):
    compound = str(compound).upper()
    if 'SOFT' in compound: return '#FF3333', 'S'
    if 'MEDIUM' in compound: return '#FFFF00', 'M'
    if 'HARD' in compound: return '#FFFFFF', 'H'
    if 'INTER' in compound: return '#00FF00', 'I'
    if 'WET' in compound: return '#0000FF', 'W'
    return '#888888', '?'

def hex_to_rgb(hex_col):
    hex_col = hex_col.lstrip('#')
    return tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))

def get_ordinal(n):
    if 11 <= (n % 100) <= 13: return f"{n}th"
    return f"{n}" + ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]

def load_fonts(base_size):
    bold = "Formula1-Bold_web_0.ttf.ttf"
    reg = "Formula1-Regular_web_0.ttf.ttf"
    
    if not os.path.exists(bold) or not os.path.exists(reg):
        print("Warning: F1 fonts not found, falling back to default.")
        d = ImageFont.load_default()
        return d, d, d, d, d

    title_font = ImageFont.truetype(bold, int(base_size * 1.5))
    axis_font = ImageFont.truetype(bold, int(base_size * 0.8))
    label_font = ImageFont.truetype(bold, int(base_size * 1.1))
    tire_font = ImageFont.truetype(bold, int(base_size * 1.0))
    gap_font = ImageFont.truetype(reg, int(base_size * 0.75))
    return title_font, axis_font, label_font, tire_font, gap_font

def create_race_timelapse(year=2026, gp='Bahrain', session_type='R',
                          save_path='output.mp4', duration=30, fps=60, portrait=True,
                          session_label='RACE'):
    
    print(f"Loading pure time data for {year} {gp} Session: {session_type}...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=False)
    
    laps_data = session.laps
    drivers = pd.unique(laps_data['Driver'])
    total_laps = int(laps_data['LapNumber'].max())
    total_drivers = len(drivers) 
    
    d_data = {}
    global_start_time = float('inf')
    global_end_time = 0.0
    
    for drv in drivers:
        drv_laps = laps_data.pick_driver(drv).dropna(subset=['LapNumber', 'Position', 'Time', 'LapStartTime'])
        if drv_laps.empty: continue
            
        lap_ends = drv_laps['Time'].dt.total_seconds().values
        lap_starts = drv_laps['LapStartTime'].dt.total_seconds().values
        
        start_time = lap_starts[0]
        times = np.insert(lap_ends, 0, start_time)
        laps = np.insert(drv_laps['LapNumber'].values, 0, 0)
        pos = np.insert(drv_laps['Position'].values, 0, drv_laps['Position'].values[0])
        compounds = np.insert(drv_laps['Compound'].values, 0, drv_laps['Compound'].values[0])
        
        global_start_time = min(global_start_time, start_time)
        global_end_time = max(global_end_time, lap_ends[-1])
        
        # DYNAMICALLY QUERY CONSTRUCTOR AND MAP TO LIBRARY
        try:
            drv_info = session.results[session.results['Abbreviation'] == drv].iloc[0]
            team_name = drv_info['TeamName']
            drv_color = get_constructor_color(team_name)
        except Exception:
            drv_color = '#888888'
        
        d_data[drv] = {
            'laps': laps, 'pos': pos, 'times': times, 'compounds': compounds,
            'color': drv_color,
            'max_time': lap_ends[-1]
        }

    canvas_w, canvas_h = (1080, 1920) if portrait else (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (canvas_w, canvas_h))
    
    base_size = (canvas_w if portrait else canvas_h) * 0.025
    title_f, axis_f, label_f, tire_f, gap_f = load_fonts(base_size)
    
    graph_offset_l = 150 
    graph_offset_t = 180 
    graph_offset_b = 100 
    graph_offset_r = 60
    
    graph_w = canvas_w - graph_offset_l - graph_offset_r
    graph_h = canvas_h - graph_offset_t - graph_offset_b
    plot_w = graph_w - 200 
    
    line_w = int(canvas_w * 0.012)
    dot_r = int(canvas_w * 0.018)
    
    total_frames = duration * fps
    print(f"Rendering {total_frames} frames to {save_path}...")
    
    for frame_idx in range(total_frames):
        progress = frame_idx / (total_frames - 1)
        current_time = global_start_time + progress * (global_end_time - global_start_time)
        
        current_states = {}
        leader_drv = max(d_data.keys(), key=lambda d: np.interp(min(current_time, d_data[d]['max_time']), d_data[d]['times'], d_data[d]['laps']))
        leader_data = d_data[leader_drv]
        
        for drv, data in d_data.items():
            eval_time = min(current_time, data['max_time'])
            cur_lap = np.interp(eval_time, data['times'], data['laps'])
            cur_pos = np.interp(eval_time, data['times'], data['pos'])
            
            leader_t_at_lap = np.interp(cur_lap, leader_data['laps'], leader_data['times'])
            gap_to_leader = max(0.0, eval_time - leader_t_at_lap)
            
            is_dnf = data['laps'][-1] < total_laps
            time_since_finish = current_time - data['max_time']
            alpha = max(0, int(255 - (time_since_finish * 5))) if (is_dnf and time_since_finish > 0) else 255
            
            current_states[drv] = {
                'lap': cur_lap, 'pos': cur_pos, 'eval_time': eval_time,
                'alpha': alpha, 'gap': gap_to_leader
            }

        active_laps = [s['lap'] for d, s in current_states.items() if s['alpha'] > 0]
        leader_lap = current_states[leader_drv]['lap']
        last_lap = min(active_laps) if active_laps else leader_lap
        
        window_size = max(1.0, (leader_lap - last_lap) * 1.15)
        view_max_x = leader_lap + (window_size * 0.1)
        view_min_x = view_max_x - window_size

        def get_x(lap_num): return ((lap_num - view_min_x) / window_size) * plot_w
        def get_y(position): 
            denominator = max(1.0, float(total_drivers - 1))
            return dot_r + 10 + ((position - 1) / denominator) * (graph_h - 2 * (dot_r + 10))

        base_img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 255))
        draw_base = ImageDraw.Draw(base_img)
        
        draw_base.rectangle([20, 20, canvas_w - 20, canvas_h - 20], outline=(255, 255, 255, 255), width=4)
        
        wm_str = "@formulytics"
        try:
            wm_bbox = label_f.getbbox(wm_str)
            draw_base.text((canvas_w/2 - (wm_bbox[2]-wm_bbox[0])/2, 110), wm_str, font=label_f, fill=(100, 100, 100, 140))
        except AttributeError: pass

        for pos in range(1, total_drivers + 1):
            y_abs = graph_offset_t + get_y(pos)
            pos_str = get_ordinal(pos)
            try:
                bbox = axis_f.getbbox(pos_str)
                draw_base.text((graph_offset_l - (bbox[2]-bbox[0]) - 20, y_abs - (bbox[3]-bbox[1])/2), pos_str, font=axis_f, fill=(200, 200, 200, 255))
            except AttributeError: pass

        tick_step = 5.0 if window_size > 10 else 1.0
        first_tick = int(view_min_x / tick_step) * tick_step
        for tick_idx in range(int(window_size / tick_step) + 3):
            tick = first_tick + (tick_idx * tick_step)
            if tick < 0: continue
            x_rel = get_x(tick)
            if 0 <= x_rel <= plot_w:
                x_abs = graph_offset_l + x_rel
                draw_base.line([(x_abs, graph_offset_t), (x_abs, graph_offset_t + graph_h)], fill=(30, 30, 30, 255), width=1)
                tick_str = str(int(tick))
                try:
                    bbox = axis_f.getbbox(tick_str)
                    draw_base.text((x_abs - (bbox[2]-bbox[0])/2, graph_offset_t + graph_h + 15), tick_str, font=axis_f, fill=(200, 200, 200, 255))
                except AttributeError: pass

        title_str = f"{gp.upper()} {year} - {session_label} - LAP {int(leader_lap)}/{total_laps}"
        cur_title_f = title_f
        cur_size = int(base_size * 1.5)
        try:
            t_bbox = cur_title_f.getbbox(title_str)
            while (t_bbox[2] - t_bbox[0]) > (canvas_w - 60):
                cur_size -= 2
                cur_title_f = ImageFont.truetype("Formula1-Bold_web_0.ttf.ttf", cur_size)
                t_bbox = cur_title_f.getbbox(title_str)
            draw_base.text((canvas_w/2 - (t_bbox[2]-t_bbox[0])/2, 50), title_str, font=cur_title_f, fill=(255, 0, 50, 255))
        except AttributeError: pass
        
        try:
            lap_title_str = "LAP"
            lap_bbox = label_f.getbbox(lap_title_str)
            draw_base.text((canvas_w/2 - (lap_bbox[2]-lap_bbox[0])/2, canvas_h - 70), lap_title_str, font=label_f, fill=(255, 255, 255, 255))
        except AttributeError: pass

        graph_img = Image.new('RGBA', (int(graph_w), int(graph_h)), (0, 0, 0, 0))
        draw_graph = ImageDraw.Draw(graph_img)

        for pos in range(1, total_drivers + 1):
            draw_graph.line([(0, get_y(pos)), (graph_w, get_y(pos))], fill=(30, 30, 30, 255), width=1)

        for drv, state in current_states.items():
            alpha = state['alpha']
            if alpha == 0: continue
            
            data = d_data[drv]
            team_rgb = hex_to_rgb(data['color'])
            line_color = (*team_rgb, alpha)
            
            pts = []
            for i, t in enumerate(data['times']):
                if t <= state['eval_time']:
                    pts.append((get_x(data['laps'][i]), get_y(data['pos'][i])))
            
            head_x = get_x(state['lap'])
            head_y = get_y(state['pos'])
            pts.append((head_x, head_y))
            
            if len(pts) > 1:
                draw_graph.line(pts, fill=line_color, width=line_w, joint="curve")
                
            draw_graph.ellipse([head_x - dot_r, head_y - dot_r, head_x + dot_r, head_y + dot_r], fill=line_color, outline=(255, 255, 255, alpha), width=max(2, int(line_w*0.4)))
            
            lbl_x = head_x + dot_r + 20
            try:
                l_bbox = label_f.getbbox(drv)
                lh, lw = l_bbox[3] - l_bbox[1], l_bbox[2] - l_bbox[0]
                draw_graph.text((lbl_x, head_y - lh/2 - 5), drv, font=label_f, fill=(255, 255, 255, alpha))
            except AttributeError:
                lw = 60
                draw_graph.text((lbl_x, head_y - 15), drv, font=label_f, fill=(255, 255, 255, alpha))

            cur_comp_idx = np.searchsorted(data['times'], state['eval_time'], side='right') - 1
            cur_comp_idx = np.clip(cur_comp_idx, 0, len(data['compounds']) - 1)
            tire_hex, tire_letter = get_tire_info(data['compounds'][cur_comp_idx])
            
            t_rad = int(dot_r * 1.5)
            t_x = lbl_x + lw + 25
            draw_graph.ellipse([t_x - t_rad, head_y - t_rad, t_x + t_rad, head_y + t_rad], fill=(*hex_to_rgb(tire_hex), alpha))
            try:
                t_bbox = tire_f.getbbox(tire_letter)
                tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
                draw_graph.text((t_x - tw/2, head_y - th/2 - 4), tire_letter, font=tire_f, fill=(0, 0, 0, alpha))
            except AttributeError: pass
                
            if drv != leader_drv and state['gap'] > 0:
                gap_str = f"+{state['gap']:.1f}"
                draw_graph.text((lbl_x, head_y + dot_r + 5), gap_str, font=gap_f, fill=(200, 200, 200, alpha))

        base_img.paste(graph_img, (graph_offset_l, graph_offset_t), graph_img)

        scaled_w = int(canvas_w * GLOBAL_SCALE)
        scaled_h = int(canvas_h * GLOBAL_SCALE)
        scaled_img = base_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        
        final_frame = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))
        paste_x = (canvas_w - scaled_w) // 2
        paste_y = (canvas_h - scaled_h) // 2
        final_frame.paste(scaled_img, (paste_x, paste_y))

        video.write(cv2.cvtColor(np.array(final_frame), cv2.COLOR_RGB2BGR))
        if frame_idx % (fps * 2) == 0: print(f"Processed {frame_idx // fps}s / {duration}s")
            
    video.release()
    print("Video rendered successfully!")


# ============================================================
# SCHEDULING & ORCHESTRATION
# ============================================================
def session_label_for(kind):
    """kind is 'R' or 'S'; returns the on-screen title label."""
    return "SPRINT" if kind == "S" else "RACE"


def session_end_time(start, kind):
    """Estimated wall-clock end of a session, given its start datetime."""
    buffer = SPRINT_END_BUFFER_MIN if kind == "S" else RACE_END_BUFFER_MIN
    return start + timedelta(minutes=buffer)


def retry_slots(end_time):
    """Datetimes at which the workflow should retry rendering this session."""
    return [end_time + timedelta(minutes=off) for off in RETRY_OFFSETS_MIN]


def cron_for_datetime(dt):
    """One UTC datetime -> a GitHub Actions 5-field cron string."""
    return f"{dt.minute} {dt.hour} {dt.day} {dt.month} *"


def output_slug(event_name, year, kind):
    """Stable output filename, e.g. 'canadian_grand_prix_2026_R.mp4'."""
    base = re.sub(r"[^a-z0-9]+", "_", event_name.lower()).strip("_")
    return f"{base}_{year}_{kind}.mp4"


def list_sessions(schedule_df):
    """FastF1 event-schedule DataFrame -> list of race/sprint session dicts.

    Matches the session names 'Race' and 'Sprint' exactly, so 'Sprint
    Qualifying' / 'Sprint Shootout' are ignored.
    """
    sessions = []
    for _, ev in schedule_df.iterrows():
        for i in range(1, 6):
            name = ev.get(f"Session{i}")
            date = ev.get(f"Session{i}DateUtc")
            if name not in ("Race", "Sprint"):
                continue
            if date is None or pd.isna(date):
                continue
            kind = "S" if name == "Sprint" else "R"
            start = pd.Timestamp(date).to_pydatetime()
            sessions.append({
                "event": ev["EventName"],
                "round": int(ev["RoundNumber"]),
                "kind": kind,
                "start": start,
                "end": session_end_time(start, kind),
            })
    return sessions


def schedulable_sessions(sessions, now):
    """Sessions worth writing cron slots for: their last retry slot is still
    in the future, and they start within the lookahead horizon."""
    last_offset = max(RETRY_OFFSETS_MIN)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)
    out = []
    for s in sessions:
        last_slot = s["end"] + timedelta(minutes=last_offset)
        if last_slot > now and s["start"] < horizon:
            out.append(s)
    return out


def due_sessions(sessions, now):
    """Sessions that have ended and are still inside the render window."""
    out = []
    for s in sessions:
        if s["end"] <= now <= s["end"] + timedelta(hours=RENDER_LOOKBACK_HOURS):
            out.append(s)
    return out


def build_cron_lines(sessions, now):
    """Sorted, de-duplicated cron strings: the weekly safety tick plus a retry
    slot for every schedulable session."""
    lines = {WEEKLY_SAFETY_CRON}
    for s in schedulable_sessions(sessions, now):
        for slot in retry_slots(s["end"]):
            lines.add(cron_for_datetime(slot))
    return sorted(lines)


def rewrite_cron_block(workflow_text, cron_lines):
    """Return workflow_text with the region between CRON_BEGIN and CRON_END
    replaced by `cron_lines`. The marker comment lines themselves are kept."""
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


def load_event_schedule(season):
    """Return the FastF1 event-schedule DataFrame for a season."""
    return fastf1.get_event_schedule(season, include_testing=False)


def session_has_data(year, event_name, kind):
    """True once FastF1 has lap data for the session.

    Returns False (so a later cron slot retries) both when the session is not
    loadable yet and when it loads but has no laps. The exception text is
    printed so a genuine, persistent failure is still visible in the logs.
    """
    try:
        session = fastf1.get_session(year, event_name, kind)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        print(f"  session not loadable yet ({event_name} {kind}): {exc}")
        return False
    if len(session.laps) == 0:
        print(f"  session loaded but has no laps yet ({event_name} {kind})")
        return False
    return True


def get_drive_service():
    """Authenticated Google Drive v3 client, using the GDRIVE_* env vars."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials(
        token=None,
        refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
        client_id=os.environ["GDRIVE_CLIENT_ID"],
        client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)


def drive_has_file(service, folder_id, filename):
    """True if a non-trashed file named `filename` exists in the folder."""
    safe = filename.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{safe}' and '{folder_id}' in parents and trashed = false"
    resp = service.files().list(q=query, fields="files(id,name)").execute()
    return len(resp.get("files", [])) > 0


def drive_upload_file(service, folder_id, path):
    """Upload a local file into the Drive folder; return the new file id."""
    from googleapiclient.http import MediaFileUpload

    metadata = {"name": os.path.basename(path), "parents": [folder_id]}
    media = MediaFileUpload(path, mimetype="video/mp4", resumable=True)
    result = service.files().create(
        body=metadata, media_body=media, fields="id,name"
    ).execute()
    return result["id"]


def commit_and_push(path, message):
    """Commit a single file and push. Only acts inside GitHub Actions; on a
    local machine it prints what it would do and returns."""
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


def parse_force_session(text):
    """'2026 Miami R' -> (2026, 'Miami', 'R'). Event name may contain spaces."""
    parts = text.split()
    if len(parts) < 3:
        raise SystemExit(
            '--force-session must be "YEAR EVENT KIND", e.g. "2026 Miami R"'
        )
    year, event, kind = int(parts[0]), " ".join(parts[1:-1]), parts[-1].upper()
    if kind not in ("R", "S"):
        raise SystemExit(f'--force-session kind must be R or S, got "{kind}"')
    return year, event, kind


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
    commit_and_push(WORKFLOW_PATH, "chore: update race-replay cron schedule")
    print("Schedule updated.")


def render_due(now):
    """Phase 2: render any just-ended session not already in Google Drive."""
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
        # session_has_data loads the session as a readiness gate;
        # create_race_timelapse loads it again to render. The second load is
        # cheap thanks to the FastF1 cache. Keep both: the readiness gate must
        # remain even if the render path changes.
        if not session_has_data(SEASON, s["event"], s["kind"]):
            print(f"  DATA_NOT_READY ({s['event']} {s['kind']})")
            continue
        out_path = os.path.join(OUTPUT_DIR, slug)
        create_race_timelapse(
            year=SEASON, gp=s["event"], session_type=s["kind"],
            save_path=out_path, duration=DURATION_SECONDS, fps=FPS,
            portrait=PORTRAIT, session_label=session_label_for(s["kind"]),
        )
        file_id = drive_upload_file(service, folder, out_path)
        print(f"  UPLOADED ({slug}, drive id {file_id})")


def main():
    parser = argparse.ArgumentParser(description="F1 race-replay pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cron block without writing or committing")
    parser.add_argument("--force-session", metavar="SPEC",
                        help='render one session locally, e.g. "2026 Miami R"')
    parser.add_argument("--duration", type=int, default=DURATION_SECONDS,
                        help="override video length in seconds (for testing)")
    args = parser.parse_args()

    if args.force_session:
        year, event, kind = parse_force_session(args.force_session)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, output_slug(event, year, kind))
        create_race_timelapse(
            year=year, gp=event, session_type=kind, save_path=out_path,
            duration=args.duration, fps=FPS, portrait=PORTRAIT,
            session_label=session_label_for(kind),
        )
        print(f"Rendered {out_path}")
        return

    now = datetime.utcnow()
    print(f"=== race_replay.py @ {now} UTC (season {SEASON}) ===")
    reschedule(now, dry_run=args.dry_run)
    if not args.dry_run:
        render_due(now)


if __name__ == "__main__":
    main()