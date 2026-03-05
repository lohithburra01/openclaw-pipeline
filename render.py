#!/usr/bin/env python
# render.py — F1 Formulytics Cloud Renderer
# Runs headless on Google Colab
#
# Colab setup (run once):
#   !pip install fastf1 opencv-python-headless pillow scipy numpy -q
#
# Usage:
#   !python render.py --config config.json

import fastf1
import fastf1.plotting
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

COLOR_MAP = {
    'red bull': '#3671C6', 'mercedes': '#27F4D2', 'ferrari': '#FF3333',
    'mclaren': '#FF8000', 'aston martin': '#229971', 'racing point': '#F596C8',
    'force india': '#F596C8', 'alpine': '#0090D0', 'renault': '#FFF500',
    'williams': '#64C4FF', 'alphatauri': '#2B4562', 'toro rosso': '#0000FF',
    'racing bulls': '#6692FF', 'rb': '#6692FF', 'alfa romeo': '#900000',
    'kick sauber': '#52E252', 'sauber': '#52E252', 'haas': '#B6BABD'
}

def hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def get_team_color(driver_code, session_obj):
    try:
        team_raw = session_obj.get_driver(driver_code)['TeamName'].lower()
        for key, hex_color in COLOR_MAP.items():
            if key in team_raw:
                return hex_to_bgr(hex_color)
    except:
        pass
    try:
        return hex_to_bgr(fastf1.plotting.driver_color(driver_code))
    except:
        return (255, 255, 255)

def calculate_hifi_delta(ref, tgt):
    master_len = ref['dist'].max()
    scale = master_len / tgt['dist'].max()
    tgt_dist_scaled = tgt['dist'] * scale

    grid_len = 10000; window_size = 300; step_size = 50
    common_grid = np.linspace(0, master_len, grid_len)
    v_ref = np.interp(common_grid, ref['dist'], ref['speed'])
    v_tgt = np.interp(common_grid, tgt_dist_scaled, tgt['speed'])

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
            shifts.append(shift_m)
            positions.append(start_pos + window_size / 2)

    if positions:
        if positions[0] > 0: positions.insert(0, 0); shifts.insert(0, shifts[0])
        if positions[-1] < master_len: positions.append(master_len); shifts.append(shifts[-1])
        positions, shifts = zip(*sorted(zip(positions, shifts)))
        shift_interp = interp1d(positions, shifts, kind='linear', fill_value="extrapolate")
        tgt_dist_warped = tgt_dist_scaled + shift_interp(tgt_dist_scaled)
    else:
        tgt_dist_warped = tgt_dist_scaled

    f_tgt_time = interp1d(tgt_dist_warped, tgt['time'], fill_value="extrapolate")
    raw_delta = f_tgt_time(ref['dist']) - ref['time']
    delta_smooth = medfilt(raw_delta, kernel_size=15)
    delta_zeroed = delta_smooth - delta_smooth[0]

    actual_lap_diff = tgt['lap_time'] - ref['lap_time']
    residual = delta_zeroed[-1] - actual_lap_diff
    if abs(residual) > 1e-6:
        final_delta = delta_zeroed - (np.linspace(0, 1, len(delta_zeroed)) * residual)
    else:
        final_delta = delta_zeroed

    return ref['dist'], final_delta

class F1VideoBaker:
    def __init__(self, track, configs, is_same_race, output_path,
                 zoom_factor=3.0, trail_frames=60, fps=30):
        self.track        = track
        self.configs      = configs
        self.is_same_race = is_same_race
        self.output_path  = output_path
        self.zoom_factor  = zoom_factor
        self.trail_frames = trail_frames
        self.fps          = fps
        self.ref_config   = configs[0]
        self.sessions     = {}
        self._load_fonts()

    def _load_fonts(self):
        font_dir = os.environ.get("FONT_DIR", "")
        try:
            self.font_title  = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 42)
            self.font_sub    = ImageFont.truetype(os.path.join(font_dir, "Formula1-Regular_web_0.ttf.ttf"), 26)
            self.font_lb_pos = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 20)
            self.font_lb_name= ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 30)
            self.font_lb_gap = ImageFont.truetype(os.path.join(font_dir, "Formula1-Regular_web_0.ttf.ttf"), 26)
            self.font_car    = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 20)
            self.font_wm     = ImageFont.truetype(os.path.join(font_dir, "Formula1-Bold_web_0.ttf.ttf"), 24)
            print("✅ Custom fonts loaded.")
        except:
            print("ℹ️  Using default fonts.")
            f = ImageFont.load_default()
            self.font_title = self.font_sub = self.font_lb_pos = \
            self.font_lb_name = self.font_lb_gap = self.font_car = self.font_wm = f

    def _get_session(self, year, session_type):
        key = (year, self.track, session_type)
        if key not in self.sessions:
            print(f"⏳ Loading: {year} {self.track} ({session_type})...")
            try: fastf1.Cache.enable_cache('cache')
            except: pass
            s = fastf1.get_session(year, self.track, session_type)
            s.load(telemetry=True, laps=True, weather=False, messages=False)
            self.sessions[key] = s
        return self.sessions[key]

    def _get_trace(self, config):
        try:
            session = self._get_session(config['year'], config['session'])
            laps = session.laps.pick_driver(config['driver'])
            if laps.empty: return None
            lap = laps.pick_fastest()
            tel = lap.get_telemetry().dropna(subset=['Distance', 'Speed', 'X', 'Y'])
            tel = tel.drop_duplicates(subset=['Time'])
            dist = tel['Distance'].values
            dist -= dist[0]
            return {
                'dist':     dist,
                'speed':    tel['Speed'].values / 3.6,
                'time':     tel['Time'].dt.total_seconds().values,
                'x':        tel['X'].values / 10.0,
                'y':        tel['Y'].values / 10.0,
                'lap_time': lap['LapTime'].total_seconds(),
                'driver':   config['driver'],
                'config':   config,
                'session':  session
            }
        except Exception as e:
            print(f"⚠️  Error loading {config['driver']} ({config['year']}): {e}")
            return None

    def _draw_centered(self, draw, x, y, text, font, fill):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except:
            w, h = draw.textsize(text, font=font)
        draw.text((x - w/2, y - h/2 - 4), text, font=font, fill=fill)

    def _display_name(self, config):
        if self.is_same_race: return config['driver']
        return f"{config['driver']} '{str(config['year'])[-2:]} ({config['session']})"

    def bake(self):
        ref_trace = self._get_trace(self.ref_config)
        if not ref_trace:
            print("❌ Reference driver trace missing. Aborting.")
            return None

        title_line1 = self.track.upper()
        if self.is_same_race: title_line1 += f" - {self.ref_config['year']}"
        title_line2 = (
            f"TOP {len(self.configs)} - {self.ref_config['session'].upper()} LAP COMPARISON"
            if self.is_same_race else "CROSS-SESSION LAP COMPARISON"
        )

        valid_traces, time_mappings, delta_functions = [], {}, {}

        print("⚙️  Running HiFi Delta Engine...")
        for cfg in self.configs:
            trace = self._get_trace(cfg)
            if not trace: continue
            uid = f"{cfg['driver']}_{cfg['year']}_{cfg['session']}"
            trace['uid'] = uid
            valid_traces.append(trace)

            if cfg == self.ref_config:
                time_mappings[uid] = np.maximum.accumulate(ref_trace['time'])
                delta_functions[uid] = lambda d: 0.0
            else:
                _, final_delta = calculate_hifi_delta(ref_trace, trace)
                mapped = ref_trace['time'] + final_delta
                time_mappings[uid] = np.maximum.accumulate(mapped)
                delta_functions[uid] = interp1d(ref_trace['dist'], final_delta, fill_value="extrapolate")

        if not valid_traces:
            print("❌ No valid traces. Aborting.")
            return None

        print("🛤️  Building track spline...")
        rx = ref_trace['x'] - np.mean(ref_trace['x'])
        ry = ref_trace['y'] - np.mean(ref_trace['y'])
        tck, _ = splprep([rx, ry], s=100, per=0)

        def get_pos(prog):
            x, y = splev(np.clip(prog, 0, 1), tck)
            return float(x), float(y)

        track_pts = [get_pos(p) for p in np.linspace(0, 1, 2000)]
        master_len = ref_trace['dist'][-1]
        max_time = max(t[-1] for t in time_mappings.values())
        total_frames = int(max_time * self.fps)
        vec_arrays = {tr['uid']: (time_mappings[tr['uid']], ref_trace['dist']) for tr in valid_traces}

        colors = {
            tr['uid']: get_team_color(tr['config']['driver'], tr['session'])
            for tr in valid_traces
        }

        WIDTH, HEIGHT = 1080, 1920
        out_dir = os.path.dirname(self.output_path)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (WIDTH, HEIGHT))

        track_w = max(p[0] for p in track_pts) - min(p[0] for p in track_pts)
        track_h = max(p[1] for p in track_pts) - min(p[1] for p in track_pts)
        scale = (min(WIDTH, HEIGHT * 0.5) / max(track_w, track_h)) * self.zoom_factor

        def w2s(x, y, cx, cy):
            return int(WIDTH/2 + (x-cx)*scale), int(HEIGHT*0.4 - (y-cy)*scale)

        MAP_SIZE = 280
        MAP_X = WIDTH - MAP_SIZE - 60
        lb_y_start = 1350 - (len(valid_traces) * 70)
        MAP_Y = 1350 - MAP_SIZE
        map_scale = MAP_SIZE / max(track_w, track_h) * 0.95

        def w2m(x, y):
            return int(MAP_X + MAP_SIZE/2 + x*map_scale), int(MAP_Y + MAP_SIZE/2 - y*map_scale)

        track_arr = np.array(track_pts)
        map_track = np.array([w2m(p[0], p[1]) for p in track_pts], np.int32)
        ref_uid = f"{self.ref_config['driver']}_{self.ref_config['year']}_{self.ref_config['session']}"
        trails = {tr['uid']: [] for tr in valid_traces}

        print(f"📼 Rendering {total_frames} frames @ {self.fps}fps...")
        for f_idx in range(total_frames):
            if f_idx % (self.fps * 10) == 0:
                print(f"   {f_idx/total_frames*100:.0f}% ({f_idx}/{total_frames})")

            t = f_idx / self.fps
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            cur_dists = {tr['uid']: np.interp(t, vec_arrays[tr['uid']][0], vec_arrays[tr['uid']][1])
                         for tr in valid_traces}
            ref_dist = cur_dists[ref_uid]
            cam_x, cam_y = get_pos(ref_dist / master_len)

            pts = np.array([w2s(p[0], p[1], cam_x, cam_y) for p in track_arr], np.int32)
            cv2.polylines(frame, [pts.reshape(-1,1,2)], False, (255,255,255), 40, cv2.LINE_AA)
            cv2.polylines(frame, [pts.reshape(-1,1,2)], False, (40,40,40),   32, cv2.LINE_AA)

            draw_order = sorted(valid_traces, key=lambda tr: cur_dists[tr['uid']])

            for tr in draw_order:
                uid = tr['uid']
                px, py = get_pos(cur_dists[uid] / master_len)
                trails[uid].append((px, py))
                if len(trails[uid]) > self.trail_frames: trails[uid].pop(0)
                if len(trails[uid]) > 1:
                    sc = np.array([w2s(tx, ty, cam_x, cam_y) for tx, ty in trails[uid]], np.int32)
                    cv2.polylines(frame, [sc.reshape(-1,1,2)], False, colors[uid], 18, cv2.LINE_AA)

            cv2.polylines(frame, [map_track.reshape(-1,1,2)], False, (80,80,80), 6, cv2.LINE_AA)
            for tr in draw_order:
                uid = tr['uid']
                px, py = get_pos(cur_dists[uid] / master_len)
                sx, sy = w2s(px, py, cam_x, cam_y)
                mx, my = w2m(px, py)
                color = colors[uid]
                cv2.circle(frame, (sx, sy), 30, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (sx, sy), 30, (255,255,255), 4, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (mx, my), 8, (255,255,255), 2, cv2.LINE_AA)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            self._draw_centered(draw, WIDTH//2, 250, title_line1, self.font_title, (255,255,255))
            self._draw_centered(draw, WIDTH//2, 300, title_line2, self.font_sub, (200,200,200))

            for tr in draw_order:
                uid = tr['uid']
                sx, sy = w2s(*get_pos(cur_dists[uid]/master_len), cam_x, cam_y)
                name = self._display_name(tr['config'])
                try:
                    bbox = draw.textbbox((0,0), name, font=self.font_car)
                    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                except:
                    tw, th = draw.textsize(name, font=self.font_car)
                ly = sy - 55
                draw.rectangle([sx-tw/2-8, ly-th/2-6, sx+tw/2+8, ly+th/2+6], fill=(0,0,0))
                self._draw_centered(draw, sx, ly, name, self.font_car, (255,255,255))

            sorted_lb = sorted(valid_traces, key=lambda tr: cur_dists[tr['uid']], reverse=True)
            for i, tr in enumerate(sorted_lb):
                uid = tr['uid']
                dist = cur_dists[uid]
                c_bgr = colors[uid]
                c_rgb = (c_bgr[2], c_bgr[1], c_bgr[0])
                cy_pos = lb_y_start + i * 70 + 15

                if uid == ref_uid:
                    gap_text, gap_color = "LEADER", (150,150,150)
                else:
                    sign = "+" if dist < ref_dist else "-"
                    time_gap = delta_functions[uid](dist)
                    gap_text = f"{sign}{abs(dist-ref_dist):.0f}m ({sign}{abs(time_gap):.3f}s)"
                    gap_color = (230,50,50) if dist < ref_dist else (50,230,50)

                draw.ellipse([62, cy_pos-18, 98, cy_pos+18], fill=c_rgb)
                self._draw_centered(draw, 80, cy_pos, f"P{i+1}", self.font_lb_pos, (255,255,255))
                name = self._display_name(tr['config'])
                draw.text((120, cy_pos-18), name, font=self.font_lb_name, fill=c_rgb)
                try: nw = draw.textbbox((0,0), name, font=self.font_lb_name)[2]
                except: nw = draw.textsize(name, font=self.font_lb_name)[0]
                draw.text((120+nw+35, cy_pos-14), gap_text, font=self.font_lb_gap, fill=gap_color)

            self._draw_centered(draw, WIDTH//2, 1850, "@formulytics", self.font_wm, (150,150,150))
            out.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        out.release()
        size_mb = os.path.getsize(self.output_path) / 1_000_000
        print(f"✅ Done: {self.output_path} ({size_mb:.1f} MB) | {total_frames} frames @ {self.fps}fps")
        return self.output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--output-dir', default='output')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    jobs = config.get("jobs", [config])
    os.makedirs(args.output_dir, exist_ok=True)

    for i, job in enumerate(jobs):
        print(f"\n{'='*50}")
        print(f"🏎️  Job {i+1}/{len(jobs)}: {job['track']}")
        print(f"{'='*50}")

        is_same_race = "configs" not in job
        configs = ([{"driver": d, "year": job["year"], "session": job["session"]}
                    for d in job["drivers"]] if is_same_race else job["configs"])

        track = job["track"]
        drivers_str = "_".join(c['driver'] for c in configs)
        safe_track = track.replace(' ', '_').lower()
        out_path = job.get("output", os.path.join(args.output_dir,
                           f"{safe_track}_{configs[0]['year']}_{configs[0]['session']}_{drivers_str}.mp4"))

        baker = F1VideoBaker(
            track=track, configs=configs, is_same_race=is_same_race,
            output_path=out_path,
            zoom_factor=job.get("zoom_factor", 3.0),
            trail_frames=job.get("trail_frames", 60),
            fps=job.get("fps", 30)
        )
        result = baker.bake()
        if result:
            print(f"📁 Saved: {result}")

if __name__ == "__main__":
    main()