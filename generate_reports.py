#!/usr/bin/env python3
"""
Generate reports from knee angle tracking CSV files.

This script analyzes CSV files containing knee angle tracking data and generates
comprehensive reports including:
- Angle plots over time for tracked knees
- Processing time analysis (if available)
- Statistical analysis of angles and tracking performance

All output files are saved in the 'reports' directory with the same base name
as the input CSV file:
- {basename}_angles.png: Plot of knee angles over time
- {basename}_timing.png: Plot of frame processing times
- {basename}_stats.txt: Comprehensive statistics report

Usage:
    python generate_reports.py <path_to_csv>

Example:
    python generate_reports.py data/knee_angles.csv
    # This will create:
    #   reports/knee_angles_angles.png
    #   reports/knee_angles_timing.png
    #   reports/knee_angles_stats.txt
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict
import matplotlib.collections

def read_csv_data(csv_path: str) -> pd.DataFrame:
    """Read and validate the CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ['frame', 'timeframe']  # Updated to match our CSV structure
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Rename timeframe to time for consistency
    df = df.rename(columns={'timeframe': 'time'})
    return df

def get_tracked_knees(df: pd.DataFrame) -> List[str]:
    """Determine which knees are tracked based on non-null values."""
    knees = []
    for knee in ['left', 'right']:
        # Check both angle and keypoint columns
        angle_col = f'{knee}_knee_angle'
        conf_col = f'{knee}_knee_conf'
        if angle_col in df.columns and conf_col in df.columns:
            if not (df[angle_col].isna().all() or df[conf_col].isna().all()):
                knees.append(knee)
    return knees

def plot_angles(df: pd.DataFrame, output_path: str, tracked_knees: List[str]):
    """Generate angle plots for tracked knees."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot left knee
    if 'left' in tracked_knees:
        if 'left_knee_incorrect' in df.columns:
            try:
                # Create a line collection for segments with different colors
                points = np.array([df['time'], df['left_knee_angle']]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = ['purple' if incorrect else 'red' for incorrect in df['left_knee_incorrect'].iloc[:-1]]
                
                lc = matplotlib.collections.LineCollection(segments, colors=colors, label='Angle')
                ax1.add_collection(lc)
                ax1.autoscale()
            except:
                # Fallback to simple angle plot if error occurs
                ax1.plot(df['time'], df['left_knee_angle'], color='red', label='Angle')
                print("Warning: Could not plot colored segments for left knee")
        else:
            ax1.plot(df['time'], df['left_knee_angle'], color='red', label='Angle')
    
    # Plot right knee
    if 'right' in tracked_knees:
        if 'right_knee_incorrect' in df.columns:
            try:
                # Create a line collection for segments with different colors
                points = np.array([df['time'], df['right_knee_angle']]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                colors = ['purple' if incorrect else 'red' for incorrect in df['right_knee_incorrect'].iloc[:-1]]
                
                lc = matplotlib.collections.LineCollection(segments, colors=colors, label='Angle')
                ax2.add_collection(lc)
                ax2.autoscale()
            except:
                # Fallback to simple angle plot if error occurs
                ax2.plot(df['time'], df['right_knee_angle'], color='blue', label='Angle')
                print("Warning: Could not plot colored segments for right knee")
        else:
            ax2.plot(df['time'], df['right_knee_angle'], color='blue', label='Angle')
    
    # Format plots
    fig.suptitle('Knee Angles Over Time (MP2D)', fontsize=14)
    ax1.set_title('Left Knee')
    ax2.set_title('Right Knee')
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        # Add custom legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='purple', lw=2),
                       Line2D([0], [0], color='red', lw=2)]
        ax.legend(custom_lines, ['Correct', 'Incorrect'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence(df: pd.DataFrame, output_path: str, tracked_knees: List[str]):
    """Generate confidence plots for tracked keypoints."""
    keypoints = ['hip', 'knee', 'ankle', 'foot_index']
    colors = ['blue', 'red', 'green', 'orange']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Function to get min/max confidence for a side
    def get_conf_range(side):
        conf_values = []
        for kp in keypoints:
            col_name = f'{side}_{kp}_conf'
            if col_name in df.columns:
                conf_values.extend(df[col_name].dropna().tolist())
        if conf_values:
            return min(conf_values), max(conf_values)
        return None, None
    
    # Plot left keypoints confidence
    if 'left' in tracked_knees:
        min_conf, max_conf = get_conf_range('left')
        if min_conf is not None:
            # Add 5% padding to the range
            padding = (max_conf - min_conf) * 0.05
            ax1.set_ylim(max(0, min_conf - padding), min(1, max_conf + padding))
            
            for kp, color in zip(keypoints, colors):
                col_name = f'left_{kp}_conf'
                if col_name in df.columns:
                    ax1.plot(df['time'], df[col_name], color=color, label=kp.replace('_', ' ').title())
    
    # Plot right keypoints confidence
    if 'right' in tracked_knees:
        min_conf, max_conf = get_conf_range('right')
        if min_conf is not None:
            # Add 5% padding to the range
            padding = (max_conf - min_conf) * 0.05
            ax2.set_ylim(max(0, min_conf - padding), min(1, max_conf + padding))
            
            for kp, color in zip(keypoints, colors):
                col_name = f'right_{kp}_conf'
                if col_name in df.columns:
                    ax2.plot(df['time'], df[col_name], color=color, label=kp.replace('_', ' ').title())
    
    # Format plots
    fig.suptitle('Keypoint Confidence Over Time (MP2D)', fontsize=14)
    ax1.set_title('Left Keypoints')
    ax2.set_title('Right Keypoints')
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Confidence')
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_timing(df: pd.DataFrame, output_path: str):
    """Generate timing plot if processing time data is available."""
    if 'frame_processing_time_ms' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Calculate moving average
    window_size = 30  # 1 second at 30fps
    times = df['frame_processing_time_ms']
    moving_avg = times.rolling(window=window_size, center=True).mean()
    
    # Calculate outliers (points more than 2 standard deviations from moving average)
    std_dev = times.std()
    outliers = times[abs(times - moving_avg) > 2 * std_dev]
    outlier_times = df['time'][outliers.index]
    
    # Plot raw data in light blue
    plt.plot(df['time'], times, 'lightblue', alpha=0.5, label='Raw')
    
    # Plot moving average in dark blue
    plt.plot(df['time'], moving_avg, 'blue', linewidth=2, label='Moving Average')
    
    # Plot outliers in red
    if not outliers.empty:
        plt.scatter(outlier_times, outliers, color='red', label='Outliers', zorder=5)
    
    plt.title('Frame Processing Time')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Processing Time (ms)')
    plt.legend()
    
    # Add text with statistics
    stats_text = f"Statistics:\n" \
                 f"Mean: {times.mean():.1f} ms\n" \
                 f"Std Dev: {std_dev:.1f} ms\n" \
                 f"Outliers: {len(outliers)} frames"
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_accuracy_metrics(df: pd.DataFrame, knee: str) -> dict:
    """Analyze accuracy metrics for a specific knee."""
    metrics = {}
    
    # Get relevant columns
    angle_col = f'{knee}_knee_angle'
    conf_col = f'{knee}_knee_conf'
    hip_conf = f'{knee}_hip_conf'
    knee_conf = f'{knee}_knee_conf'
    ankle_conf = f'{knee}_ankle_conf'
    
    if angle_col not in df.columns:
        return None
        
    angles = df[angle_col].dropna()
    
    # 1. Keypoint Detection Confidence
    if hip_conf in df.columns and knee_conf in df.columns and ankle_conf in df.columns:
        conf_data = {
            'hip': df[hip_conf].mean(),
            'knee': df[knee_conf].mean(),
            'ankle': df[ankle_conf].mean(),
            'min_conf': min(df[hip_conf].min(), df[knee_conf].min(), df[ankle_conf].min()),
            'missing_rate': (df[[hip_conf, knee_conf, ankle_conf]] < 0.5).any(axis=1).mean() * 100
        }
        metrics['confidence'] = conf_data
    
    # 2. Keypoint Stability (jitter between frames)
    if hip_conf in df.columns:
        # Calculate frame-to-frame changes in position
        for point in ['hip', 'knee', 'ankle']:
            x_col = f'{knee}_{point}_x'
            y_col = f'{knee}_{point}_y'
            if x_col in df.columns and y_col in df.columns:
                dx = df[x_col].diff().abs()
                dy = df[y_col].diff().abs()
                movement = np.sqrt(dx**2 + dy**2)
                metrics[f'{point}_jitter'] = {
                    'mean': movement.mean(),
                    'max': movement.max(),
                    'std': movement.std()
                }
    
    # 3. Angle Analysis
    if not angles.empty:
        # Basic angle stats
        metrics['angle_stats'] = {
            'mean': angles.mean(),
            'std': angles.std(),
            'min': angles.min(),
            'max': angles.max(),
            'range': angles.max() - angles.min()
        }
        
        # Smoothness (angle changes between frames)
        angle_velocity = angles.diff().abs()
        metrics['angle_smoothness'] = {
            'mean_velocity': angle_velocity.mean(),
            'max_velocity': angle_velocity.max(),
            'velocity_std': angle_velocity.std()
        }
        
        # Detect outliers using z-score
        z_scores = np.abs((angles - angles.mean()) / angles.std())
        outliers = z_scores > 2
        metrics['angle_outliers'] = {
            'count': outliers.sum(),
            'percentage': (outliers.sum() / len(angles)) * 100,
            'max_z_score': z_scores.max()
        }
        
        # Angle consistency (using rolling statistics)
        window = 30  # 1 second at 30fps
        rolling_mean = angles.rolling(window=window, center=True).mean()
        rolling_std = angles.rolling(window=window, center=True).std()
        metrics['angle_consistency'] = {
            'mean_std': rolling_std.mean(),
            'max_std': rolling_std.max(),
            'trend_changes': np.diff(np.signbit(np.diff(rolling_mean))).sum()
        }
    
    return metrics

def generate_stats_report(df: pd.DataFrame, output_path: str, tracked_knees: List[str]):
    """Generate comprehensive statistics report."""
    with open(output_path, 'w') as f:
        f.write("Statistics Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Processing Statistics
        f.write("Processing Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total frames: {len(df)}\n")
        duration = df['time'].max() - df['time'].min()
        f.write(f"Video duration: {duration:.2f} seconds\n")
        f.write(f"Average FPS: {len(df) / duration:.2f}\n")
        
        if 'frame_processing_time_ms' in df.columns:
            proc_times = df['frame_processing_time_ms']
            f.write(f"Processing time (ms):\n")
            f.write(f"  Average: {proc_times.mean():.2f}\n")
            f.write(f"  Min: {proc_times.min():.2f}\n")
            f.write(f"  Max: {proc_times.max():.2f}\n")
            f.write(f"  Std dev: {proc_times.std():.2f}\n")
        
        # Accuracy Metrics for each knee
        for knee in ['left', 'right']:
            if knee in tracked_knees:
                f.write(f"\n{knee.capitalize()} Knee Analysis:\n")
                f.write("-" * 30 + "\n")
                
                metrics = analyze_accuracy_metrics(df, knee)
                if metrics:
                    # Confidence
                    if 'confidence' in metrics:
                        f.write("\nKeypoint Detection Confidence:\n")
                        conf = metrics['confidence']
                        f.write(f"  Hip: {conf['hip']:.3f}\n")
                        f.write(f"  Knee: {conf['knee']:.3f}\n")
                        f.write(f"  Ankle: {conf['ankle']:.3f}\n")
                        f.write(f"  Minimum confidence: {conf['min_conf']:.3f}\n")
                        f.write(f"  Missing keypoint rate: {conf['missing_rate']:.1f}%\n")
                    
                    # Stability
                    for point in ['hip', 'knee', 'ankle']:
                        key = f'{point}_jitter'
                        if key in metrics:
                            f.write(f"\n{point.capitalize()} Stability:\n")
                            jitter = metrics[key]
                            f.write(f"  Average movement: {jitter['mean']:.3f}\n")
                            f.write(f"  Max movement: {jitter['max']:.3f}\n")
                            f.write(f"  Movement std: {jitter['std']:.3f}\n")
                    
                    # Angle Analysis
                    if 'angle_stats' in metrics:
                        f.write("\nAngle Statistics:\n")
                        stats = metrics['angle_stats']
                        f.write(f"  Mean angle: {stats['mean']:.2f}°\n")
                        f.write(f"  Angle std: {stats['std']:.2f}°\n")
                        f.write(f"  Range: {stats['range']:.2f}° ({stats['min']:.1f}° - {stats['max']:.1f}°)\n")
                        
                        smooth = metrics['angle_smoothness']
                        f.write("\nAngle Smoothness:\n")
                        f.write(f"  Average velocity: {smooth['mean_velocity']:.2f}°/frame\n")
                        f.write(f"  Max velocity: {smooth['max_velocity']:.2f}°/frame\n")
                        f.write(f"  Velocity std: {smooth['velocity_std']:.2f}°/frame\n")
                        
                        outliers = metrics['angle_outliers']
                        f.write("\nAngle Outliers:\n")
                        f.write(f"  Count: {outliers['count']} frames\n")
                        f.write(f"  Percentage: {outliers['percentage']:.1f}%\n")
                        f.write(f"  Max deviation: {outliers['max_z_score']:.1f} std\n")
                        
                        consist = metrics['angle_consistency']
                        f.write("\nAngle Consistency:\n")
                        f.write(f"  Average variation: {consist['mean_std']:.2f}°\n")
                        f.write(f"  Max variation: {consist['max_std']:.2f}°\n")
                        f.write(f"  Trend changes: {consist['trend_changes']}\n")
            else:
                f.write(f"\n{knee.capitalize()} Knee: Not tracked\n")

def generate_reports(csv_path: str):
    """Main function to generate all reports from a CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Read data
    df = read_csv_data(csv_path)
    tracked_knees = get_tracked_knees(df)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Generate reports
    plot_angles(df, os.path.join('reports', f'{base_name}_angles.png'), tracked_knees)
    plot_confidence(df, os.path.join('reports', f'{base_name}_confidence.png'), tracked_knees)
    plot_timing(df, os.path.join('reports', f'{base_name}_timing.png'))
    generate_stats_report(df, os.path.join('reports', f'{base_name}_stats.txt'), tracked_knees)
    
    print(f"Reports generated in the 'reports' directory for {base_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_reports.py <path_to_csv>")
        sys.exit(1)
    
    generate_reports(sys.argv[1]) 