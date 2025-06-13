import pandas as pd
import sys
import re
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python stats_report.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

if df.empty or (df.isna().all(axis=None)):
    print("No data in CSV file.")
    sys.exit(0)

# Helper to find best matching column

def find_col(target, columns):
    # Normalize: lowercase, remove underscores, spaces, and parentheses
    norm = lambda s: re.sub(r'[^a-z0-9]', '', s.lower())
    target_norm = norm(target)
    for col in columns:
        if norm(col) == target_norm:
            return col
    # Try partial match
    for col in columns:
        if target_norm in norm(col):
            return col
    return None

knee_sides = ['left', 'right']

for side in knee_sides:
    print(f"\n--- {side.capitalize()} Knee Stats ---")
    conf_targets = [f'{side}_hip_conf', f'{side}_knee_conf', f'{side}_ankle_conf', f'{side}_foot_index_conf']
    angle_target = f'{side}_knee_angle'
    incorrect_target = f'{side}_knee_incorrect'
    # Map to actual columns
    conf_cols = [find_col(t, df.columns) for t in conf_targets]
    angle_col = find_col(angle_target, df.columns)
    incorrect_col = find_col(incorrect_target, df.columns)
    # Warn if mapping
    for t, c in zip(conf_targets, conf_cols):
        if c is None:
            print(f"  Warning: Could not find column for '{t}'")
        elif t != c:
            print(f"  Info: Using column '{c}' for '{t}'")
    if angle_col is None:
        print(f"  Warning: Could not find column for '{angle_target}'")
    elif angle_col != angle_target:
        print(f"  Info: Using column '{angle_col}' for '{angle_target}'")
    if incorrect_col is None:
        print(f"  Warning: Could not find column for '{incorrect_target}'")
    elif incorrect_col != incorrect_target:
        print(f"  Info: Using column '{incorrect_col}' for '{incorrect_target}'")
    # Check if there is any non-NaN data for this knee
    has_data = any((c in df and df[c].notna().any()) for c in conf_cols if c)
    if not has_data:
        print(f"{side.capitalize()} knee: Not tracked")
        continue
    # Average confidence for each keypoint
    for t, col in zip(conf_targets, conf_cols):
        if col and col in df:
            valid = pd.to_numeric(df[col], errors='coerce').dropna()
            if not valid.empty:
                avg_conf = valid.mean()
                print(f"Average {col}: {avg_conf:.3f}")
            else:
                print(f"Average {col}: N/A")
        else:
            print(f"Average {t}: N/A")
    # Percentage of frames with all keypoints detected (conf > 0.5)
    if all(c for c in conf_cols):
        detected = (pd.to_numeric(df[conf_cols[0]], errors='coerce') > 0.5) & \
                   (pd.to_numeric(df[conf_cols[1]], errors='coerce') > 0.5) & \
                   (pd.to_numeric(df[conf_cols[2]], errors='coerce') > 0.5) & \
                   (pd.to_numeric(df[conf_cols[3]], errors='coerce') > 0.5)
        if detected.notna().any():
            percent_detected = detected.mean() * 100
            print(f"Frames with all {side} keypoints detected (>0.5): {percent_detected:.1f}%")
        else:
            print(f"Frames with all {side} keypoints detected (>0.5): N/A")
    else:
        print(f"Frames with all {side} keypoints detected (>0.5): N/A")
    # Average knee angle
    if angle_col and angle_col in df:
        valid = pd.to_numeric(df[angle_col], errors='coerce').dropna()
        if not valid.empty:
            avg_angle = valid.mean()
            print(f"Average {angle_col}: {avg_angle:.2f}")
        else:
            print(f"Average {angle_col}: N/A")
    else:
        print(f"Average {angle_target}: N/A")
    # Incorrect execution rate
    if incorrect_col and incorrect_col in df:
        valid = pd.to_numeric(df[incorrect_col], errors='coerce').dropna()
        if not valid.empty:
            incorrect_rate = (valid == 0).mean() * 100
            print(f"Incorrect {side} knee execution rate: {incorrect_rate:.1f}%")
        else:
            print(f"Incorrect {side} knee execution rate: N/A")
    else:
        print(f"Incorrect {side} knee execution rate: N/A")

# FPS estimate
frame_col = find_col('frame', df.columns)
time_col = find_col('timeframe', df.columns)
if frame_col and time_col and len(df) > 1:
    valid_time = pd.to_numeric(df[time_col], errors='coerce').dropna()
    valid_frame = pd.to_numeric(df[frame_col], errors='coerce').dropna()
    if len(valid_time) > 1 and len(valid_frame) > 1:
        total_frames = valid_frame.max() - valid_frame.min() + 1
        total_time = valid_time.iloc[-1] - valid_time.iloc[0]
        fps = total_frames / total_time if total_time > 0 else None
        print(f"\nEstimated FPS: {fps:.2f}" if fps else "\nFPS could not be calculated")
    else:
        print("\nFPS could not be calculated: Not enough valid data (need at least 2 frames with valid time and frame numbers)")
else:
    print("\nFPS could not be calculated: Missing or insufficient 'frame' or 'timeframe' columns")

# Frame processing time stats
proc_time_col = find_col('frame_processing_time_ms', df.columns)
if proc_time_col and proc_time_col in df:
    proc_times = pd.to_numeric(df[proc_time_col], errors='coerce').dropna()
    if not proc_times.empty:
        print(f"\n--- Frame Processing Time Stats (ms) ---")
        print(f"Average: {proc_times.mean():.2f} ms")
        print(f"Min: {proc_times.min():.2f} ms")
        print(f"Max: {proc_times.max():.2f} ms")
        print(f"Std Dev: {proc_times.std():.2f} ms")
        print(f"10 Fastest Frames: {proc_times.nsmallest(10).values}")
        print(f"10 Slowest Frames: {proc_times.nlargest(10).values}")
    else:
        print("No frame processing time data available.")
else:
    print("No frame processing time column found in CSV.") 