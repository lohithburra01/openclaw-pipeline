import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import cv2
import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d, splprep, splev
from scipy.signal import correlate, medfilt
import warnings

warnings.simplefilter(action='ignore')

# ==========================================
# 0. DATABASE CONNECTION
# ==========================================
DB_DIR = os.environ.get("DB_DIR", "database")
CALENDAR_DB = {}
DRIVERS_DB = {}
SEASON_DB = {}

try:
    with open(os.path.join(DB_DIR, "calendar_cache.json"), 'r', encoding='utf-8') as f:
        CALENDAR_DB = json.load(f)
    with open(os.path.join(DB_DIR, "drivers_by_race.json"), 'r', encoding='utf-8') as f:
        DRIVERS_DB = json.load(f)
    with open(os.path.join(DB_DIR, "drivers_by_season.json"), 'r', encoding='utf-8') as f:
        SEASON_DB = json.load(f)
    print(f"✅ Successfully linked local F1 database.")
except Exception as e:
    print(f"⚠️ Warning: Could not load local databases. Error: {e}")

# ==========================================
# 1. HIGH-FIDELITY DELTA ENGINE (PURE MATH)
# ==========================================
def calculate_hifi_delta(ref_trace, tgt_trace):
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
        if end_pos > master_len: break
        i0 = int((start_pos / master_len) * grid_len)
        i1 = int((end_pos / master_len) * grid_len)
        corr = correlate(v_ref[i0:i1], v_tgt[i0:i1], mode='same')
        if len(corr) == 0: continue
        lag_idx = np.argmax(corr) - (len(corr) // 2)
        shift_m = lag_idx * (master_len / grid_len)
        if abs(shift_m) < 40:
            shifts.append(shift_m); positions.append(start_pos + window_size/2)

    if positions:
        if positions[0] > 0: positions.insert(0, 0); shifts.insert(0, shifts[0])
        if positions[-1] < master_len: positions.append(master_len); shifts.append(shifts[-1])
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

# ==========================================
# 2. THE VIDEO ENGINE (CV2 EXPORTER)
# ==========================================

def draw_centered(draw, x, y, text, font, fill):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    except:
        w, h = draw.textsize(text, font=font)
    draw.text((x - w/2, y - h/2 - 4), text, font=font, fill=fill)

class F1VideoBaker:
    def __init__(self, track, configs, is_same_race=True, zoom_factor=3.0, trail_frames=60, fps=30):
        self.track = track
        self.configs = configs
        self.is_same_race = is_same_race
        self.fps = fps
        self.zoom_factor = zoom_factor
        self.trail_frames = trail_frames
        self.ref_config = configs[0]
        self.loaded_sessions = {}
        
        font_dir = ""
        try:
            self.font_title = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 42)
            self.font_sub = ImageFont.truetype(os.path.join(font_dir, "Formula1-Regular_web_0.ttf.ttf"), 26)
            self.font_lb_pos = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 20)
            self.font_lb_name = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 30)
            self.font_lb_gap = ImageFont.truetype(os.path.join(font_dir, "Formula1-Regular_web_0.ttf.ttf"), 26)
            self.font_car = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 20)
            self.font_wm = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 24)
        except:
            print("⚠️ Font load error. Using defaults.")
            def_font = ImageFont.load_default()
            self.font_title = self.font_sub = self.font_lb_pos = self.font_lb_name = self.font_lb_gap = self.font_car = self.font_wm = def_font

    def _hex_to_bgr(self, hex_str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6: return (255, 255, 255)
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def get_team_colors(self, driver_code, year):
        team_raw = ""
        if str(year) in DRIVERS_DB and self.track in DRIVERS_DB[str(year)]:
            race_data = DRIVERS_DB[str(year)][self.track]
            entries = []
            if isinstance(race_data, dict):
                for v in race_data.values():
                    if isinstance(v, list):
                        entries.extend(v)
            elif isinstance(race_data, list):
                entries = race_data
            for d in entries:
                if isinstance(d, dict) and d.get('code') == driver_code:
                    team_raw = d.get('team_raw', '').lower()
                    break
        if not team_raw and str(year) in SEASON_DB:
            for d in SEASON_DB[str(year)]:
                if d['code'] == driver_code:
                    team_raw = d.get('team_raw', '').lower()
                    break
                    
        color_map = {
            'red bull': '#3671C6', 'mercedes': '#27F4D2', 'ferrari': '#FF3333',
            'mclaren': '#FF8000', 'aston martin': '#229971', 'racing point': '#F596C8',
            'force india': '#F596C8', 'alpine': '#0090D0', 'renault': '#FFF500',
            'williams': '#64C4FF', 'alphatauri': '#2B4562', 'toro rosso': '#0000FF',
            'racing bulls': '#6692FF', 'rb': '#6692FF', 'alfa romeo': '#900000',
            'kick sauber': '#52E252', 'sauber': '#52E252', 'haas': '#B6BABD'
        }
        for key, hex_color in color_map.items():
            if key in team_raw: return self._hex_to_bgr(hex_color)
            
        try: return self._hex_to_bgr(fastf1.plotting.driver_color(driver_code))
        except: return (255, 255, 255)

    def get_session(self, year, session_type):
        key = (year, self.track, session_type)
        if key not in self.loaded_sessions:
            print(f"⏳ Loading Session: {year} {self.track} ({session_type})...")
            try: fastf1.Cache.enable_cache('cache')
            except: pass
            s = fastf1.get_session(year, self.track, session_type)
            s.load(telemetry=True, laps=True, weather=False, messages=False)
            self.loaded_sessions[key] = s
        return self.loaded_sessions[key]

    def get_clean_trace(self, config):
        try:
            session = self.get_session(config['year'], config['session'])
            laps = session.laps.pick_driver(config['driver'])
            if laps.empty: return None
            lap = laps.pick_fastest()
            tel = lap.get_telemetry().dropna(subset=['Distance', 'Speed', 'X', 'Y'])
            tel = tel.drop_duplicates(subset=['Time'])
            
            dist = tel['Distance'].values
            dist -= dist[0]
            
            return {
                'dist': dist,
                'speed': tel['Speed'].values / 3.6,
                'time': tel['Time'].dt.total_seconds().values,
                'x': tel['X'].values / 10.0,
                'y': tel['Y'].values / 10.0,
                'lap_time': lap['LapTime'].total_seconds(),
                'driver': config['driver'],
                'config': config
            }
        except Exception as e:
            print(f"⚠️ Error with {config['driver']} ({config['year']}): {e}")
            return None

    def format_display_name(self, config):
        if self.is_same_race: return config['driver']
        yr = str(config['year'])[-2:]
        return f"{config['driver']} '{yr} ({config['session']})"

    def bake(self, output_path=None):
        ref_trace = self.get_clean_trace(self.ref_config)
        if not ref_trace:
            print("❌ Reference driver trace missing. Cannot build track spline.")
            return

        title_line1 = f"{self.track.upper()}"
        if self.is_same_race: title_line1 += f" - {self.ref_config['year']}"
        title_line2 = f"TOP {len(self.configs)} - {self.ref_config['session'].upper()} LAP COMPARISON" if self.is_same_race else f"CROSS-SESSION LAP COMPARISON"
        
        master_len = ref_trace['dist'][-1]
        time_mappings = {}
        delta_functions = {}

        print("⚙️ Applying HiFi Delta Engine (Cross-Year Aligned)...")
        valid_traces = []
        for cfg in self.configs:
            tgt_trace = self.get_clean_trace(cfg)
            if not tgt_trace: continue
            
            uid = f"{cfg['driver']}_{cfg['year']}_{cfg['session']}"
            tgt_trace['uid'] = uid
            valid_traces.append(tgt_trace)
            
            if cfg == self.ref_config:
                mapped_time = ref_trace['time']
                delta_functions[uid] = lambda d: 0.0
            else:
                _, final_delta = calculate_hifi_delta(ref_trace, tgt_trace)
                mapped_time = ref_trace['time'] + final_delta
                delta_functions[uid] = interp1d(ref_trace['dist'], final_delta, fill_value="extrapolate")
                
            time_mappings[uid] = np.maximum.accumulate(mapped_time)

        print("🛤️ Generating Base Rail from Reference...")
        ref_x = ref_trace['x']
        ref_y = ref_trace['y']
        cx, cy = np.mean(ref_x), np.mean(ref_y)
        ref_x -= cx
        ref_y -= cy
        
        tck, _ = splprep([ref_x, ref_y], s=100, per=0)
        def get_pos(prog):
            p = np.clip(prog, 0, 1)
            x, y = splev(p, tck)
            return float(x), float(y)

        track_pts_normalized = [get_pos(p) for p in np.linspace(0, 1, 2000)]
        max_time = max(t[-1] for t in time_mappings.values())
        total_frames = int(max_time * self.fps)

        vec_arrays = {tr['uid']: (time_mappings[tr['uid']], ref_trace['dist']) for tr in valid_traces}

        print("📼 Rendering High-Speed MP4...")
        WIDTH, HEIGHT = 1080, 1920

        if output_path is None:
            output_path = f"f1_{self.track.replace(' ', '_').lower()}_comparison.mp4"

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (WIDTH, HEIGHT))

        track_w = max(p[0] for p in track_pts_normalized) - min(p[0] for p in track_pts_normalized)
        track_h = max(p[1] for p in track_pts_normalized) - min(p[1] for p in track_pts_normalized)
        scale = (min(WIDTH, HEIGHT * 0.5) / max(track_w, track_h)) * self.zoom_factor

        def world_to_screen(x, y, cam_x, cam_y):
            sx = int(WIDTH/2 + (x - cam_x) * scale)
            sy = int(HEIGHT*0.4 - (y - cam_y) * scale)
            return sx, sy

        track_screen_base = np.array(track_pts_normalized)
        trails = {tr['uid']: [] for tr in valid_traces}
        car_radius = 30 

        MAP_SIZE = 280
        MAP_X = WIDTH - MAP_SIZE - 60
        lb_y_start = 1350 - (len(valid_traces) * 70) 
        MAP_Y = 1350 - MAP_SIZE 
        map_scale = MAP_SIZE / max(track_w, track_h) * 0.95

        def world_to_minimap(x, y):
            return int(MAP_X + MAP_SIZE/2 + x * map_scale), int(MAP_Y + MAP_SIZE/2 - y * map_scale)

        map_track_screen = np.array([world_to_minimap(p[0], p[1]) for p in track_pts_normalized], np.int32)
        ref_uid = f"{self.ref_config['driver']}_{self.ref_config['year']}_{self.ref_config['session']}"

        for f in range(total_frames):
            if f % (self.fps * 10) == 0:
                print(f"   {f/total_frames*100:.0f}% ({f}/{total_frames})")

            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            t = f / self.fps

            current_distances = {}
            for tr in valid_traces:
                t_arr, d_arr = vec_arrays[tr['uid']]
                current_distances[tr['uid']] = np.interp(t, t_arr, d_arr)

            ref_dist = current_distances[ref_uid]
            cam_x, cam_y = get_pos(ref_dist / master_len)

            pts = np.array([world_to_screen(p[0], p[1], cam_x, cam_y) for p in track_screen_base], np.int32)
            cv2.polylines(frame, [pts.reshape((-1, 1, 2))], False, (255, 255, 255), thickness=40, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [pts.reshape((-1, 1, 2))], False, (40, 40, 40), thickness=32, lineType=cv2.LINE_AA)

            draw_order = sorted(valid_traces, key=lambda tr: current_distances[tr['uid']])
            
            for tr in draw_order:
                uid = tr['uid']
                dist = current_distances[uid]
                px, py = get_pos(dist / master_len)
                color_bgr = self.get_team_colors(tr['config']['driver'], tr['config']['year'])
                
                trails[uid].append((px, py))
                if len(trails[uid]) > self.trail_frames: trails[uid].pop(0)
                    
                if len(trails[uid]) > 1:
                    screen_trail = np.array([world_to_screen(tx, ty, cam_x, cam_y) for tx, ty in trails[uid]], np.int32)
                    cv2.polylines(frame, [screen_trail.reshape((-1, 1, 2))], False, color_bgr, thickness=18, lineType=cv2.LINE_AA)

            cv2.polylines(frame, [map_track_screen.reshape((-1, 1, 2))], False, (80, 80, 80), thickness=6, lineType=cv2.LINE_AA)
            for tr in draw_order:
                dist = current_distances[tr['uid']]
                px, py = get_pos(dist / master_len)
                sx, sy = world_to_screen(px, py, cam_x, cam_y)
                mx, my = world_to_minimap(px, py)
                color_bgr = self.get_team_colors(tr['config']['driver'], tr['config']['year'])

                cv2.circle(frame, (sx, sy), car_radius, color_bgr, -1, cv2.LINE_AA)
                cv2.circle(frame, (sx, sy), car_radius, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), 8, color_bgr, -1, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), 8, (255, 255, 255), 2, cv2.LINE_AA)

            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            draw_centered(draw, WIDTH//2, 250, title_line1, self.font_title, (255, 255, 255))
            draw_centered(draw, WIDTH//2, 300, title_line2, self.font_sub, (200, 200, 200))

            for tr in draw_order:
                uid = tr['uid']
                sx, sy = world_to_screen(*get_pos(current_distances[uid] / master_len), cam_x, cam_y)
                display_name = self.format_display_name(tr['config'])
                
                try: bbox = draw.textbbox((0, 0), display_name, font=self.font_car); tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
                except: tw, th = draw.textsize(display_name, font=self.font_car)
                    
                ly = sy - 55
                draw.rectangle([sx - tw/2 - 8, ly - th/2 - 6, sx + tw/2 + 8, ly + th/2 + 6], fill=(0,0,0))
                draw_centered(draw, sx, ly, display_name, self.font_car, (255, 255, 255))

            sorted_by_pos = sorted(valid_traces, key=lambda tr: current_distances[tr['uid']], reverse=True)
            
            for i, tr in enumerate(sorted_by_pos):
                uid = tr['uid']
                dist = current_distances[uid]
                c_bgr = self.get_team_colors(tr['config']['driver'], tr['config']['year'])
                color_rgb = (c_bgr[2], c_bgr[1], c_bgr[0]) 
                
                cy = lb_y_start + i * 70 + 15
                dist_to_ref = abs(dist - ref_dist)
                
                if uid == ref_uid:
                    gap_text = "LEADER"
                    color_text_rgb = (150, 150, 150)
                else:
                    sign = "+" if dist < ref_dist else "-"
                    time_gap = delta_functions[uid](dist)
                    gap_text = f"{sign}{dist_to_ref:.0f}m ({sign}{abs(time_gap):.3f}s)"
                    color_text_rgb = (230, 50, 50) if dist < ref_dist else (50, 230, 50)

                draw.ellipse([80-18, cy-18, 80+18, cy+18], fill=color_rgb)
                draw_centered(draw, 80, cy, f"P{i+1}", self.font_lb_pos, (255, 255, 255))
                
                display_name = self.format_display_name(tr['config'])
                draw.text((120, cy - 18), display_name, font=self.font_lb_name, fill=color_rgb)
                
                try: name_w = draw.textbbox((0, 0), display_name, font=self.font_lb_name)[2]
                except: name_w = draw.textsize(display_name, font=self.font_lb_name)[0]
                    
                draw.text((120 + name_w + 35, cy - 14), gap_text, font=self.font_lb_gap, fill=color_text_rgb)

            draw_centered(draw, WIDTH//2, 1450, "@formulytics", self.font_wm, (150, 150, 150))
            out.write(cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

        out.release()
        print(f"✅ SUCCESS: Video saved as '{output_path}'")
        return output_path


# ==========================================
# 3. CONFIG-DRIVEN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--output-dir', default='output', help='Output directory for MP4s')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = config.get("jobs", [config])

    for i, job in enumerate(jobs):
        print(f"\n{'='*50}")
        print(f"🏎️  Job {i+1}/{len(jobs)}: {job['track']}")
        print(f"{'='*50}")

        is_same_race = "configs" not in job
        if is_same_race:
            configs = [{"driver": d, "year": job["year"], "session": job["session"]}
                       for d in job["drivers"]]
        else:
            configs = job["configs"]

        track = job["track"]
        drivers_str = "_".join(c['driver'] for c in configs)
        safe_track = track.replace(' ', '_').lower()
        out_path = job.get("output", os.path.join(
            args.output_dir,
            f"{safe_track}_{configs[0]['year']}_{configs[0]['session']}_{drivers_str}.mp4"
        ))

        baker = F1VideoBaker(
            track=track,
            configs=configs,
            is_same_race=is_same_race,
            zoom_factor=job.get("zoom_factor", 3.0),
            trail_frames=job.get("trail_frames", 60),
            fps=int(job.get("fps", 30) * job.get("speed_multiplier", 1.0))
        )
        baker.bake(output_path=out_path)