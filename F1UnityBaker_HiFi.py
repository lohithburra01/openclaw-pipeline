#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastf1
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, splprep, splev, PchipInterpolator
from scipy.signal import correlate, medfilt, savgol_filter, find_peaks
import json
import ipywidgets as widgets
from IPython.display import display
import os
import warnings
warnings.simplefilter(action='ignore')

# ==========================================
# 1. HIFI DELTA ENGINE (Gold Standard)
# ==========================================

FILTER_WINDOW_M = 50
ANCHOR_STEP_M = 100

def calculate_hifi_delta(ref, tgt):
    metric_log = []

    # PHASE 1: ALIGNMENT
    master_len = ref['dist'].max()
    scale = master_len / tgt['dist'].max()
    tgt_dist_scaled = tgt['dist'] * scale

    len_diff_pct = abs(1.0 - scale) * 100.0
    metric_log.append(f"Scale: {len_diff_pct:.2f}%")

    # Windowed MSE shift alignment
    window_size = 300
    step_size = 50
    grid_len = 10000
    common_grid = np.linspace(0, master_len, grid_len)
    v_ref = np.interp(common_grid, ref['dist'], ref['speed'])
    v_tgt = np.interp(common_grid, tgt_dist_scaled, tgt['speed'])

    shifts = []
    positions = []
    for start_pos in range(0, int(master_len), step_size):
        end_pos = start_pos + window_size
        if end_pos > master_len:
            break
        idx_start = int((start_pos / master_len) * grid_len)
        idx_end = int((end_pos / master_len) * grid_len)
        corr = correlate(v_ref[idx_start:idx_end], v_tgt[idx_start:idx_end], mode='same')
        if len(corr) == 0:
            continue
        lag_idx = np.argmax(corr) - (len(corr) // 2)
        shift_m = lag_idx * (master_len / grid_len)
        if abs(shift_m) < 40:
            shifts.append(shift_m)
            positions.append(start_pos + window_size / 2)

    if positions:
        if positions[0] > 0:
            positions.insert(0, 0)
            shifts.insert(0, shifts[0])
        if positions[-1] < master_len:
            positions.append(master_len)
            shifts.append(shifts[-1])
        sorted_pairs = sorted(zip(positions, shifts))
        positions, shifts = zip(*sorted_pairs)
        shift_interp = interp1d(positions, shifts, kind='linear', fill_value="extrapolate")
        tgt_dist_warped = tgt_dist_scaled + shift_interp(tgt_dist_scaled)
    else:
        tgt_dist_warped = tgt_dist_scaled

    # Raw delta
    f_tgt_time = interp1d(tgt_dist_warped, tgt['time'], fill_value="extrapolate")
    tgt_time_mapped = f_tgt_time(ref['dist'])
    raw_delta = tgt_time_mapped - ref['time']
    delta_smooth = medfilt(raw_delta, kernel_size=15)
    delta_zeroed = delta_smooth - delta_smooth[0]

    # PHASE 2: SMART UNBEND
    uniform_dist = np.linspace(0, master_len, num=len(ref['dist']))
    error_uniform = np.interp(uniform_dist, ref['dist'], delta_zeroed)

    points_per_m = len(ref['dist']) / master_len
    window_pts = int(FILTER_WINDOW_M * points_per_m)
    if window_pts % 2 == 0:
        window_pts += 1
    if window_pts < 5:
        window_pts = 5

    smoothed = savgol_filter(error_uniform, window_length=window_pts, polyorder=3)
    peaks, _ = find_peaks(smoothed, distance=window_pts)
    valleys, _ = find_peaks(-smoothed, distance=window_pts)
    base_grid = np.arange(0, master_len, ANCHOR_STEP_M)
    all_anchors = np.unique(np.sort(np.concatenate([
        [0], base_grid,
        uniform_dist[peaks] if len(peaks) else [],
        uniform_dist[valleys] if len(valleys) else [],
        [master_len]
    ])))
    anchor_vals = np.interp(all_anchors, uniform_dist, smoothed)
    anchor_vals[0] = 0.0
    actual_lap_diff = tgt['lap_time'] - ref['lap_time']
    anchor_vals[-1] = delta_zeroed[-1] - actual_lap_diff
    f_correction = PchipInterpolator(all_anchors, anchor_vals)
    correction_curve = f_correction(ref['dist'])
    unbent_delta = delta_zeroed - correction_curve
    correction_magnitude = np.mean(np.abs(correction_curve))
    metric_log.append(f"Correction Avg: {correction_magnitude:.3f}s")

    # PHASE 3: FINAL VERIFICATION
    residual = unbent_delta[-1] - actual_lap_diff
    if abs(residual) > 1e-6:
        ramp = np.linspace(0, 1, len(unbent_delta))
        final_delta = unbent_delta - (ramp * residual)
    else:
        final_delta = unbent_delta

    integrity_score = max(0.0, 100.0 - (len_diff_pct * 10.0 + correction_magnitude * 100.0))
    print(f"   -> {tgt['driver']} [{integrity_score:.0f}%]: [{', '.join(metric_log)}]")

    return ref['dist'], final_delta, integrity_score


# ==========================================
# 2. BAKER
# ==========================================

class F1UnityBaker:
    def __init__(self, year, gp, session, drivers, fps=60):
        self.year = year
        self.gp = gp
        self.session_type = session
        self.drivers = drivers
        self.fps = fps
        self.ref_driver = drivers[0]

    def get_physics_data(self, session, driver):
        try:
            laps = session.laps.pick_driver(driver)
            lap = laps.pick_fastest()
            tel = lap.get_telemetry().dropna(subset=['Distance', 'Speed', 'X', 'Y'])

            x = tel['X'].values / 10.0
            y = tel['Y'].values / 10.0
            dist = tel['Distance'].values
            dist -= dist[0]
            speed_kmh = tel['Speed'].values
            speed_ms = speed_kmh / 3.6

            dt = np.diff(dist, prepend=0) / np.maximum(speed_ms, 1.0)
            time = np.cumsum(dt)
            official = lap['LapTime'].total_seconds()
            time = time * (official / time[-1])

            f_prog = interp1d(time, dist / dist[-1], kind='linear',
                              bounds_error=False, fill_value="extrapolate")
            f_speed = interp1d(time, speed_kmh, kind='linear',
                               bounds_error=False, fill_value="extrapolate")

            return {
                'func': f_prog,
                'speed_func': f_speed,
                'x': x,
                'y': y,
                'dist': dist,
                'speed': speed_kmh,
                'time': time,
                'lap_time': official,
                'driver': driver
            }
        except Exception as e:
            print(f" Error with {driver}: {e}")
            return None

    def bake(self):
        print(f" Loading {self.year} {self.gp} {self.session_type}...")
        session = fastf1.get_session(self.year, self.gp, self.session_type)
        session.load(telemetry=True, laps=True, weather=False, messages=False)

        drivers_data = {}
        max_time = 0
        for driver in self.drivers:
            data = self.get_physics_data(session, driver)
            if data:
                drivers_data[driver] = data
                if data['lap_time'] > max_time:
                    max_time = data['lap_time']

        if self.ref_driver not in drivers_data:
            print(" Reference driver missing")
            return

        # Build spline rail from reference driver
        print("  Generating Rail...")
        ref = drivers_data[self.ref_driver]
        ref_x = ref['x'].copy()
        ref_y = ref['y'].copy()
        cx, cy = np.mean(ref_x), np.mean(ref_y)
        ref_x -= cx
        ref_y -= cy
        tck, u = splprep([ref_x, ref_y], s=100, per=0)

        def get_pos(prog):
            p = np.clip(prog, 0, 1)
            x, y = splev(p, tck)
            return float(x), float(y)

        track_pts = [{"x": px, "y": 0, "z": py}
                     for px, py in (get_pos(p) for p in np.linspace(0, 1, 2000))]

        # Pre-calculate HiFi deltas for all non-ref drivers
        print("  Calculating HiFi Deltas...")
        delta_funcs = {}
        for driver in self.drivers:
            if driver == self.ref_driver or driver not in drivers_data:
                continue
            dist_axis, delta_vals, score = calculate_hifi_delta(
                drivers_data[self.ref_driver],
                drivers_data[driver]
            )
            # Interpolator: ref distance -> delta in seconds
            delta_funcs[driver] = interp1d(
                dist_axis, delta_vals,
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )

        # Bake frames
        print("  Baking Frames...")
        frames = []
        total_frames = int(max_time * self.fps)

        for f in range(total_frames):
            t = f / self.fps
            frame_data = {"t": round(t, 3), "cars": []}

            ref_prog = float(drivers_data[self.ref_driver]['func'](t))
            ref_dist_at_t = ref_prog * drivers_data[self.ref_driver]['dist'][-1]

            for driver in self.drivers:
                if driver not in drivers_data:
                    continue
                prog = float(drivers_data[driver]['func'](t))
                px, py = get_pos(prog)
                speed = float(drivers_data[driver]['speed_func'](t))

                if driver == self.ref_driver:
                    gap_str = ""
                else:
                    delta = float(delta_funcs[driver](ref_dist_at_t))
                    sign = "+" if delta > 0 else ""
                    gap_str = f"{sign}{delta:.3f}s"

                frame_data["cars"].append({
                    "name": driver,
                    "pos": {"x": px, "y": 0, "z": py},
                    "gap_text": gap_str,
                    "speed": round(speed, 1)
                })

            frames.append(frame_data)

        # Write JSON
        out_path = getattr(self, 'output_path', 'f1_unity_data.json')  # ✅ define first
        if os.path.exists(out_path):
            os.remove(out_path)

        export = {"track": track_pts, "frames": frames, "fps": self.fps,
                "reference": self.ref_driver}
        with open(out_path, "w") as f:
            json.dump(export, f)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f" Done. {out_path} ({size_mb:.1f} MB) | {total_frames} frames @ {self.fps}fps")


# ==========================================
# 3. CLI ENTRY POINT (Unity calls this)
# ==========================================
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',    type=int,   required=True)
    parser.add_argument('--gp',      type=str,   required=True)
    parser.add_argument('--session', type=str,   required=True)
    parser.add_argument('--drivers', type=str,   required=True)  # "VER,NOR,PIA"
    parser.add_argument('--fps',     type=int,   default=60)
    parser.add_argument('--output',  type=str,   required=True)  # full path to StreamingAssets
    args = parser.parse_args()

    fastf1.Cache.enable_cache(r'C:\Users\91910\Downloads\cache')

    drivers = [d.strip() for d in args.drivers.split(',') if d.strip()]
    baker = F1UnityBaker(
        year=args.year,
        gp=args.gp,
        session=args.session,
        drivers=drivers,
        fps=args.fps
    )

    # Override output path
    baker.output_path = args.output
    baker.bake()

