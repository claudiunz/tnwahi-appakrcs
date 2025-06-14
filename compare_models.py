import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json
from typing import Dict, List, Tuple, Optional
import cv2
import argparse
from scipy.signal import savgol_filter, correlate
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class ModelComparator:
    def __init__(self, model_files: Dict[str, str]):
        """Initialize the model comparator with mapping of models to their CSV files.
        
        Args:
            model_files: Dictionary mapping model names to their CSV file paths
        """
        self.model_files = model_files
        self.models = list(model_files.keys())
        self.metrics = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for specified models from CSV files."""
        data = {}
        frame_counts = {}
        time_ranges = {}
        
        for model, filepath in self.model_files.items():
            try:
                df = pd.read_csv(filepath)
                print(f"\nDiagnostics for {model}:")
                print(f"- Original frame count: {len(df)}")
                print(f"- Time column name: {'time' if 'time' in df.columns else 'timeframe' if 'timeframe' in df.columns else 'Not found'}")
                
                # Check time column and normalize if needed
                if 'timeframe' in df.columns:
                    df = df.rename(columns={'timeframe': 'time'})
                
                if 'time' in df.columns:
                    print(f"- Time range: {df['time'].min():.2f} to {df['time'].max():.2f} seconds")
                    time_ranges[model] = (df['time'].min(), df['time'].max())
                else:
                    print("Warning: No time column found")
                
                # Store frame count for comparison
                frame_counts[model] = len(df)
                data[model] = df
                
            except FileNotFoundError:
                print(f"No data found for {model} at {filepath}")
                continue
        
        # Print frame count comparison
        if frame_counts:
            print("\nFrame count comparison:")
            for model, count in frame_counts.items():
                print(f"{model}: {count} frames")
            
            # Check if time ranges are consistent
            if time_ranges:
                print("\nTime range comparison:")
                for model, (start, end) in time_ranges.items():
                    print(f"{model}: {start:.2f}s to {end:.2f}s (duration: {end-start:.2f}s)")
        
        return data
        
    def analyze_2d_vs_3d(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Compare 2D vs 3D model accuracy."""
        metrics = {
            'angle_stability': {},
            'keypoint_confidence': {},
            'detection_rate': {},
            'execution_accuracy': {}
        }
        
        # Compare MMPose 2D vs 3D
        if 'mmpose2d' in data and 'mmpose3d' in data:
            df_2d = data['mmpose2d']
            df_3d = data['mmpose3d']
            
            # Angle stability (standard deviation of angles)
            metrics['angle_stability']['2d'] = {
                'left': df_2d['left_knee_angle'].std(),
                'right': df_2d['right_knee_angle'].std()
            }
            metrics['angle_stability']['3d'] = {
                'left': df_3d['left_knee_angle'].std(),
                'right': df_3d['right_knee_angle'].std()
            }
            
            # Average confidence scores
            metrics['keypoint_confidence']['2d'] = {
                'left': df_2d['left_knee_conf'].mean(),
                'right': df_2d['right_knee_conf'].mean()
            }
            metrics['keypoint_confidence']['3d'] = {
                'left': df_3d['left_knee_conf'].mean(),
                'right': df_3d['right_knee_conf'].mean()
            }
            
            # Detection rate (percentage of frames with valid detections)
            metrics['detection_rate']['2d'] = {
                'left': (df_2d['left_knee_conf'] > 0.5).mean(),
                'right': (df_2d['right_knee_conf'] > 0.5).mean()
            }
            metrics['detection_rate']['3d'] = {
                'left': (df_3d['left_knee_conf'] > 0.5).mean(),
                'right': (df_3d['right_knee_conf'] > 0.5).mean()
            }
            
        return metrics
    
    def analyze_knee_symmetry(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze left vs right knee symmetry for each model."""
        symmetry_metrics = {}
        
        for model, df in data.items():
            if 'left_knee_angle' not in df.columns or 'right_knee_angle' not in df.columns:
                continue
                
            # Calculate angle differences
            angle_diff = abs(df['left_knee_angle'] - df['right_knee_angle'])
            
            symmetry_metrics[model] = {
                'mean_difference': angle_diff.mean(),
                'max_difference': angle_diff.max(),
                'std_difference': angle_diff.std(),
                'correlation': df['left_knee_angle'].corr(df['right_knee_angle'])
            }
            
        return symmetry_metrics
    
    def analyze_incorrect_executions(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze incorrect execution detection rates and false positives/negatives."""
        execution_metrics = {}
        
        for model, df in data.items():
            if 'left_knee_correct' not in df.columns or 'right_knee_correct' not in df.columns:
                continue
                
            # Calculate metrics for each knee
            for side in ['left', 'right']:
                col = f'{side}_knee_correct'
                
                # Basic metrics
                total_frames = len(df)
                incorrect_frames = (df[col] == 0).sum()
                correct_frames = (df[col] == 1).sum()
                
                execution_metrics[f"{model}_{side}"] = {
                    'incorrect_rate': incorrect_frames / total_frames,
                    'correct_rate': correct_frames / total_frames,
                    'total_frames': total_frames
                }
                
        return execution_metrics
    
    def _plot_model_comparison(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create side-by-side comparison plots for all models."""
        plt.figure(figsize=(15, 6))
        
        # Define colors for each model
        model_colors = {
            'mediapipe 2d': '#2ca02c',    # green
            'mediapipe 3d': '#17becf',    # cyan
            'mmpose 2d': '#ff7f0e',       # orange
            'mmpose 3d': '#e377c2',       # pink
            'yolo 2d': '#8c564b'          # brown
        }
        
        # Plot left knee angles
        plt.subplot(1, 2, 1)
        for model, df in data.items():
            if 'left_knee_angle' in df.columns:
                x_values = df['time'] if 'time' in df.columns else range(len(df))
                plt.plot(x_values, df['left_knee_angle'], 
                        label=model.upper(),
                        color=model_colors.get(model.lower(), '#000000'),
                        alpha=0.7)
        plt.title('Left Knee Angles - All Models')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
        
        # Plot right knee angles
        plt.subplot(1, 2, 2)
        for model, df in data.items():
            if 'right_knee_angle' in df.columns:
                x_values = df['time'] if 'time' in df.columns else range(len(df))
                plt.plot(x_values, df['right_knee_angle'], 
                        label=model.upper(),
                        color=model_colors.get(model.lower(), '#000000'),
                        alpha=0.7)
        plt.title('Right Knee Angles - All Models')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_comparison.png')
        plt.close()
    
    def _plot_stability_heatmaps(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Generate a single 2x2 subplot image with one heatmap per model for keypoint stability."""
        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for idx, (model, df) in enumerate(data.items()):
            if 'left_knee_conf' not in df.columns:
                continue
            # Calculate stability scores (rolling standard deviation)
            window_size = 10
            stability_data = pd.DataFrame()
            for joint in ['knee', 'hip', 'ankle']:
                for side in ['left', 'right']:
                    col = f'{side}_{joint}_conf'
                    if col in df.columns:
                        stability_data[f'{side}_{joint}'] = df[col].rolling(window_size).std()
            # Plot heatmap in the corresponding subplot
            sns.heatmap(stability_data.T, cmap='YlOrRd', xticklabels=False, ax=axes[idx])
            axes[idx].set_title(f"{model} Keypoint Stability")
            axes[idx].set_xlabel("Frame")
            axes[idx].set_ylabel("Keypoint")
        plt.tight_layout()
        plt.savefig(output_path / 'combined_stability_heatmap_2x2.png')
        plt.close()
    
    def _plot_angle_distributions(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create angle distribution plots."""
        plt.figure(figsize=(15, 10))
        
        for i, (model, df) in enumerate(data.items(), 1):
            plt.subplot(2, 2, i)
            if 'left_knee_angle' in df.columns:
                sns.kdeplot(data=df['left_knee_angle'], label='Left Knee', color='#9467bd')  # Purple for left
            if 'right_knee_angle' in df.columns:
                sns.kdeplot(data=df['right_knee_angle'], label='Right Knee', color='#1f77b4')  # Blue for right
            plt.title(f"{model} Angle Distribution")
            plt.xlabel("Angle (degrees)")
            plt.ylabel("Density")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'angle_distributions.png')
        plt.close()
    
    def _plot_statistics_bars(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Generate line plots comparing key statistics across models."""
        # Collect metrics for each model
        metrics = {
            'mean_angles': {'Left': [], 'Right': []},
            'angle_std': {'Left': [], 'Right': []},
            'mean_conf': {'Left': [], 'Right': []}
        }
        
        model_names = []
        
        for model, df in data.items():
            model_names.append(model)
            
            # Mean angles
            if 'left_knee_angle' in df.columns:
                metrics['mean_angles']['Left'].append(df['left_knee_angle'].mean())
            else:
                metrics['mean_angles']['Left'].append(0)
            if 'right_knee_angle' in df.columns:
                metrics['mean_angles']['Right'].append(df['right_knee_angle'].mean())
            else:
                metrics['mean_angles']['Right'].append(0)
            
            # Angle standard deviations
            if 'left_knee_angle' in df.columns:
                metrics['angle_std']['Left'].append(df['left_knee_angle'].std())
            else:
                metrics['angle_std']['Left'].append(0)
            if 'right_knee_angle' in df.columns:
                metrics['angle_std']['Right'].append(df['right_knee_angle'].std())
            else:
                metrics['angle_std']['Right'].append(0)
            
            # Mean confidence
            left_conf = []
            right_conf = []
            for part in ['hip', 'knee', 'ankle']:
                left_col = f'left_{part}_conf'
                right_col = f'right_{part}_conf'
                if left_col in df.columns:
                    left_conf.append(df[left_col].mean())
                if right_col in df.columns:
                    right_conf.append(df[right_col].mean())
            
            metrics['mean_conf']['Left'].append(np.mean(left_conf) if left_conf else 0)
            metrics['mean_conf']['Right'].append(np.mean(right_conf) if right_conf else 0)
        
        # Create subplots for each metric
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Define colors for each model
        colors = {
            'mediapipe 2d': '#1f77b4',  # Blue
            'mediapipe 3d': '#2ca02c',  # Green
            'mmpose 2d': '#ff7f0e',     # Orange
            'mmpose 3d': '#d62728'      # Red
        }
        
        # Create line plots
        metric_configs = [
            (ax1, metrics['mean_angles'], 'Mean Knee Angle', 'Degrees'),
            (ax2, metrics['angle_std'], 'Angle Standard Deviation', 'Degrees'),
            (ax3, metrics['mean_conf'], 'Mean Keypoint Confidence', 'Confidence Score')
        ]
        
        for ax, metric_data, title, ylabel in metric_configs:
            # Plot lines for each model
            x = np.arange(len(model_names))
            for model_name, model_color in colors.items():
                if model_name in model_names:
                    model_idx = model_names.index(model_name)
                    ax.plot([model_name], [metric_data['Left'][model_idx]], 'o-', 
                           color=model_color, label=model_name.upper(), markersize=8)
                    ax.plot([model_name], [metric_data['Right'][model_idx]], 's-', 
                           color=model_color, linestyle='--', alpha=0.7)
            
            # Add labels and formatting
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Set y-axis limit for confidence plot
            if title == 'Mean Keypoint Confidence':
                ax.set_ylim(0, 1)
            
            # Add legend
            if ax == ax1:  # Only add legend to first subplot
                handles = []
                labels = []
                # Add model entries
                for model_name in colors:
                    if model_name in model_names:
                        handles.append(plt.Line2D([0], [0], color=colors[model_name], marker='o', linestyle='-', label=model_name.upper()))
                # Add side entries
                handles.extend([
                    plt.Line2D([0], [0], color='gray', marker='o', linestyle='-', label='Left'),
                    plt.Line2D([0], [0], color='gray', marker='s', linestyle='--', label='Right')
                ])
                ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('Statistical Metrics Comparison Across Models', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_angle_correlation(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Plot correlation between confidence scores and angle accuracy."""
        plt.figure(figsize=(15, 10))
        
        for i, (model, df) in enumerate(data.items(), 1):
            if 'left_knee_conf' not in df.columns or 'left_knee_angle' not in df.columns:
                continue
                
            plt.subplot(2, 2, i)
            plt.scatter(df['left_knee_conf'], df['left_knee_angle'], alpha=0.5, label='Left Knee')
            plt.scatter(df['right_knee_conf'], df['right_knee_angle'], alpha=0.5, label='Right Knee')
            plt.title(f"{model} Confidence vs Angle")
            plt.xlabel("Confidence Score")
            plt.ylabel("Angle (degrees)")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_angle_correlation.png')
        plt.close()

    def _align_sequences(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align sequences from different models using correlation-based alignment."""
        from scipy.signal import correlate, savgol_filter
        import numpy as np
        
        aligned_data = {}
        reference_model = None
        reference_data = None
        
        # Find the model with the most complete data to use as reference
        for model, df in data.items():
            if 'left_knee_angle' in df.columns and 'right_knee_angle' in df.columns:
                if reference_model is None or len(df) > len(reference_data):
                    reference_model = model
                    reference_data = df
        
        if reference_model is None:
            print("No suitable reference model found")
            return data
            
        # Smooth the reference signals
        ref_left = savgol_filter(reference_data['left_knee_angle'].values, 15, 3)
        ref_right = savgol_filter(reference_data['right_knee_angle'].values, 15, 3)
        
        # Store the reference model's data
        aligned_data[reference_model] = reference_data
        
        # Align other models to the reference
        for model, df in data.items():
            if model == reference_model:
                continue
                
            if 'left_knee_angle' not in df.columns or 'right_knee_angle' not in df.columns:
                aligned_data[model] = df
                continue
            
            # Smooth the target signals
            target_left = savgol_filter(df['left_knee_angle'].values, 15, 3)
            target_right = savgol_filter(df['right_knee_angle'].values, 15, 3)
            
            # Find the best alignment offset using cross-correlation
            corr_left = correlate(ref_left, target_left, mode='full')
            corr_right = correlate(ref_right, target_right, mode='full')
            
            # Get the lag with maximum correlation for both knees
            lag_left = np.argmax(corr_left) - (len(target_left) - 1)
            lag_right = np.argmax(corr_right) - (len(target_right) - 1)
            
            # Use the average lag
            lag = int((lag_left + lag_right) / 2)
            
            # Create aligned dataframe
            aligned_df = pd.DataFrame()
            
            # Shift the data based on the lag
            if lag >= 0:
                # Target signal needs to be shifted right
                for col in df.columns:
                    values = df[col].values
                    aligned_values = np.pad(values, (lag, 0), mode='edge')[:-lag]
                    aligned_df[col] = aligned_values[:len(reference_data)]
            else:
                # Target signal needs to be shifted left
                for col in df.columns:
                    values = df[col].values
                    aligned_values = np.pad(values, (0, -lag), mode='edge')[-lag:]
                    aligned_df[col] = aligned_values[:len(reference_data)]
            
            aligned_data[model] = aligned_df
            
        return aligned_data

    def _plot_overlaid_angles(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create overlaid angle plots for each knee, comparing all models."""
        # First align the sequences
        aligned_data = self._align_sequences(data)
        
        # Define colors for each model
        model_colors = {
            'mediapipe 2d': '#2ca02c',    # green
            'mediapipe 3d': '#17becf',    # cyan
            'mmpose 2d': '#ff7f0e',       # orange
            'mmpose 3d': '#e377c2',       # pink
            'yolo 2d': '#8c564b'          # brown
        }
        
        # Create separate plots for left and right knees
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot left knee angles
        ax1.set_title('Left Knee Angles - All Models (Aligned)', fontsize=14)
        for model, df in aligned_data.items():
            if 'left_knee_angle' in df.columns:
                ax1.plot(df['left_knee_angle'], 
                        label=model,
                        color=model_colors.get(model.lower(), '#000000'),
                        alpha=0.7)
        
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Angle (degrees)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot right knee angles
        ax2.set_title('Right Knee Angles - All Models (Aligned)', fontsize=14)
        for model, df in aligned_data.items():
            if 'right_knee_angle' in df.columns:
                ax2.plot(df['right_knee_angle'], 
                        label=model,
                        color=model_colors.get(model.lower(), '#000000'),
                        alpha=0.7)
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Angle (degrees)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'overlaid_angles_aligned.png')
        plt.close()

    def _plot_incorrect_executions(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create plots showing incorrect executions over time for each model."""
        plt.figure(figsize=(15, 6))
        
        # Plot left knee incorrect executions
        plt.subplot(1, 2, 1)
        for model, df in data.items():
            if 'left_knee_incorrect' in df.columns and 'left_knee_angle' in df.columns:
                x_values = df['time'] if 'time' in df.columns else df['timeframe'] if 'timeframe' in df.columns else range(len(df))
                # Plot the full angle curve in light purple
                plt.plot(x_values, df['left_knee_angle'], 
                        color='#9467bd', alpha=0.3, label=f'{model} Angle')
                
                # Highlight incorrect sections in red (1 means incorrect)
                incorrect_mask = df['left_knee_incorrect'].astype(bool)  # True means incorrect
                incorrect_angles = df['left_knee_angle'].copy()
                incorrect_angles[~incorrect_mask] = np.nan  # Set correct points to NaN
                plt.plot(x_values, incorrect_angles, 
                        color='red', alpha=0.7, 
                        label=f'{model} Incorrect' if incorrect_mask.any() else None)
        
        plt.title('Left Knee Angles - Incorrect Executions')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
        
        # Plot right knee incorrect executions
        plt.subplot(1, 2, 2)
        for model, df in data.items():
            if 'right_knee_incorrect' in df.columns and 'right_knee_angle' in df.columns:
                x_values = df['time'] if 'time' in df.columns else df['timeframe'] if 'timeframe' in df.columns else range(len(df))
                # Plot the full angle curve in light blue
                plt.plot(x_values, df['right_knee_angle'], 
                        color='#1f77b4', alpha=0.3, label=f'{model} Angle')
                
                # Highlight incorrect sections in red (1 means incorrect)
                incorrect_mask = df['right_knee_incorrect'].astype(bool)  # True means incorrect
                incorrect_angles = df['right_knee_angle'].copy()
                incorrect_angles[~incorrect_mask] = np.nan  # Set correct points to NaN
                plt.plot(x_values, incorrect_angles, 
                        color='red', alpha=0.7,
                        label=f'{model} Incorrect' if incorrect_mask.any() else None)
        
        plt.title('Right Knee Angles - Incorrect Executions')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'incorrect_executions.png')
        plt.close()

    def _plot_error_detection(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create stacked bar charts showing when each model detected incorrect executions."""
        plt.figure(figsize=(15, 6))
        
        # Define colors for each model
        model_colors = {
            'mediapipe 2d': '#2ca02c',    # green
            'mediapipe 3d': '#17becf',    # cyan
            'mmpose 2d': '#ff7f0e',       # orange
            'mmpose 3d': '#e377c2',       # pink
            'yolo 2d': '#8c564b'          # brown
        }
        
        # Plot left knee error detection
        plt.subplot(1, 2, 1)
        bottom = np.zeros(len(next(iter(data.values()))))  # Initialize bottom of stack
        
        for model, df in data.items():
            if 'left_knee_incorrect' in df.columns:
                x_values = df['time'] if 'time' in df.columns else df['timeframe'] if 'timeframe' in df.columns else range(len(df))
                # Invert the boolean values - show bars only for incorrect executions
                incorrect = (~df['left_knee_incorrect'].astype(bool)).astype(int)
                plt.bar(x_values, incorrect, bottom=bottom, 
                       label=model, color=model_colors.get(model.lower(), '#000000'),
                       alpha=0.7, width=0.1)
                bottom += incorrect
        
        plt.title('Left Knee Error Detection (Stacked)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Models Detecting Error')
        plt.grid(True, axis='y')
        plt.legend()
        
        # Plot right knee error detection
        plt.subplot(1, 2, 2)
        bottom = np.zeros(len(next(iter(data.values()))))  # Initialize bottom of stack
        
        for model, df in data.items():
            if 'right_knee_incorrect' in df.columns:
                x_values = df['time'] if 'time' in df.columns else df['timeframe'] if 'timeframe' in df.columns else range(len(df))
                # Invert the boolean values - show bars only for incorrect executions
                incorrect = (~df['right_knee_incorrect'].astype(bool)).astype(int)
                plt.bar(x_values, incorrect, bottom=bottom,
                       label=model, color=model_colors.get(model.lower(), '#000000'),
                       alpha=0.7, width=0.1)
                bottom += incorrect
        
        plt.title('Right Knee Error Detection (Stacked)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Models Detecting Error')
        plt.grid(True, axis='y')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'error_detection.png')
        plt.close()

    def _plot_confidence_timeline(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Plot confidence timeline for each model."""
        plt.figure(figsize=(15, 10))
        
        for i, (model, df) in enumerate(data.items(), 1):
            plt.subplot(2, 2, i)
            if 'left_knee_conf' in df.columns:
                plt.plot(df['left_knee_conf'], label='Left Knee', color='#9467bd')  # Purple for left
            if 'right_knee_conf' in df.columns:
                plt.plot(df['right_knee_conf'], label='Right Knee', color='#1f77b4')  # Blue for right
            plt.title(f"{model} Confidence Timeline")
            plt.xlabel("Frame")
            plt.ylabel("Confidence Score")
            plt.legend()
            # Set y-axis limits with a check for NaN/Inf
            left_min_conf = df['left_knee_conf'].min() if 'left_knee_conf' in df.columns else 0
            left_max_conf = df['left_knee_conf'].max() if 'left_knee_conf' in df.columns else 1
            right_min_conf = df['right_knee_conf'].min() if 'right_knee_conf' in df.columns else 0
            right_max_conf = df['right_knee_conf'].max() if 'right_knee_conf' in df.columns else 1
            min_conf = min(left_min_conf, right_min_conf)
            max_conf = max(left_max_conf, right_max_conf)
            if np.isnan(min_conf) or np.isinf(min_conf):
                min_conf = 0
            if np.isnan(max_conf) or np.isinf(max_conf):
                max_conf = 1
            plt.ylim(min_conf - 0.1, max_conf + 0.1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_timeline.png')
        plt.close()

    def calculate_fps(self, df: pd.DataFrame) -> float:
        """Calculate FPS from frame and timeframe data."""
        frame_col = [col for col in df.columns if 'frame' in col.lower() and 'time' not in col.lower() and 'processing' not in col.lower()]
        time_col = [col for col in df.columns if 'timeframe' in col.lower()]
        
        if frame_col and time_col:
            frame_col = frame_col[0]
            time_col = time_col[0]
            
            valid_time = pd.to_numeric(df[time_col], errors='coerce').dropna()
            valid_frame = pd.to_numeric(df[frame_col], errors='coerce').dropna()
            
            if len(valid_time) > 1 and len(valid_frame) > 1:
                total_frames = valid_frame.max() - valid_frame.min() + 1
                total_time = valid_time.iloc[-1] - valid_time.iloc[0]
                return total_frames / total_time if total_time > 0 else 0
        
        # Try using frame_processing_time if available
        proc_time_col = [col for col in df.columns if 'frame_processing_time_ms' in col.lower()]
        if proc_time_col:
            proc_times = pd.to_numeric(df[proc_time_col[0]], errors='coerce').dropna()
            if not proc_times.empty:
                # Convert ms to seconds and calculate FPS
                avg_time_sec = proc_times.mean() / 1000
                return 1 / avg_time_sec if avg_time_sec > 0 else 0
        
        return 0

    def _plot_inference_speed(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create visualization showing inference speed over time."""
        # Define colors for each model
        model_colors = {
            'mediapipe 2d': '#2ca02c',    # green
            'mediapipe 3d': '#17becf',    # cyan
            'mmpose 2d': '#ff7f0e',       # orange
            'mmpose 3d': '#e377c2',       # pink
            'yolo 2d': '#8c564b'          # brown
        }
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        has_data = False
        
        # Calculate FPS over time for each model
        for model_name, df in data.items():
            model_key = model_name.lower()
            
            # Try to get processing time data
            proc_time_col = None
            for col in df.columns:
                if 'frame_processing_time' in col.lower():
                    proc_time_col = col
                    break
            
            if proc_time_col is not None:
                # Convert processing time to numeric, handling any errors
                proc_times = pd.to_numeric(df[proc_time_col], errors='coerce')
                proc_times = proc_times[proc_times > 0]  # Remove any zero or negative values
                
                if not proc_times.empty:
                    # Convert processing time (ms) to FPS
                    fps_series = 1000 / proc_times  # Convert ms to FPS
                    
                    # Apply rolling mean to smooth the line (if we have enough data points)
                    if len(fps_series) > 10:
                        fps_series = fps_series.rolling(window=10, min_periods=1).mean()
                    
                    # Plot FPS over time using frame numbers
                    color = model_colors.get(model_key, '#000000')
                    plt.plot(range(len(fps_series)), fps_series, 
                            label=model_key.upper(), 
                            color=color, 
                            linewidth=2)
                    has_data = True
        
        if not has_data:
            print("No valid processing time data found in any model's data")
            plt.close()
            return
        
        # Customize plot
        plt.title('Inference Speed Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Frames Per Second (FPS)')
        plt.grid(True, alpha=0.3)
        
        # Only add legend if we have data
        if has_data:
            plt.legend(loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path / 'inference_speed.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _output_fps_stats(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create a table showing average FPS statistics for each model."""
        fps_stats = {}
        
        for model_name, df in data.items():
            model_key = model_name.lower()
            
            # Try to get processing time data
            proc_time_col = None
            for col in df.columns:
                if 'frame_processing_time' in col.lower():
                    proc_time_col = col
                    break
            
            if proc_time_col is not None:
                # Convert processing time to numeric, handling any errors
                proc_times = pd.to_numeric(df[proc_time_col], errors='coerce')
                proc_times = proc_times[proc_times > 0]  # Remove any zero or negative values
                
                if not proc_times.empty:
                    # Calculate FPS statistics
                    fps_values = 1000 / proc_times  # Convert ms to FPS
                    fps_stats[model_key] = {
                        'mean': fps_values.mean(),
                        'std': fps_values.std(),
                        'min': fps_values.min(),
                        'max': fps_values.max()
                    }
        
        if fps_stats:
            # Create DataFrame for better formatting
            stats_df = pd.DataFrame.from_dict(fps_stats, orient='index')
            
            # Round values to 2 decimal places
            stats_df = stats_df.round(2)
            
            # Sort by mean FPS (descending)
            stats_df = stats_df.sort_values('mean', ascending=False)
            
            # Save to CSV
            stats_df.to_csv(output_path / 'fps_stats.csv')
            
            # Also create a formatted text file
            with open(output_path / 'fps_stats.txt', 'w') as f:
                f.write("Average Inference Speed (FPS) for Each Model\n")
                f.write("==========================================\n\n")
                
                # Calculate column widths
                model_width = max(len("Model"), max(len(model) for model in fps_stats.keys()))
                
                # Create header
                header = f"{'Model':<{model_width}} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}\n"
                separator = f"{'-' * model_width} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8}\n"
                
                f.write(header)
                f.write(separator)
                
                # Write each row
                for model in stats_df.index:
                    row = stats_df.loc[model]
                    f.write(f"{model.upper():<{model_width}} | {row['mean']:8.2f} | {row['std']:8.2f} | {row['min']:8.2f} | {row['max']:8.2f}\n")

    def _output_angle_stats(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create a table showing knee angle statistics for each model."""
        angle_stats = {}
        print("\nDebug: Processing knee angle statistics")
        
        for model_name, df in data.items():
            model_key = model_name.lower()
            print(f"\nModel: {model_name}")
            print(f"Available columns: {df.columns.tolist()}")
            
            model_stats = {'left': {}, 'right': {}}
            
            # Process left and right knee angles
            for side in ['left', 'right']:
                angle_col = f'{side}_knee_angle'
                print(f"Looking for column: {angle_col}")
                if angle_col in df.columns:
                    print(f"Found {angle_col}")
                    angles = pd.to_numeric(df[angle_col], errors='coerce')
                    angles = angles.dropna()  # Remove NaN values
                    
                    if not angles.empty:
                        print(f"Processing {len(angles)} valid angle values")
                        model_stats[side] = {
                            'mean': angles.mean(),
                            'std': angles.std(),
                            'min': angles.min(),
                            'max': angles.max(),
                            'range': angles.max() - angles.min()
                        }
                else:
                    print(f"Column {angle_col} not found")
            
            # Only include model if we have stats for at least one knee
            if model_stats['left'] or model_stats['right']:
                angle_stats[model_key] = model_stats
        
        if angle_stats:
            print("\nGenerating knee angle statistics files...")
            # Create a formatted text file
            with open(output_path / 'knee_angle_stats.txt', 'w') as f:
                f.write("Summary of Knee Angle Statistics for Each Model\n")
                f.write("==========================================\n\n")
                
                # Calculate column widths
                model_width = max(len("Model"), max(len(model) for model in angle_stats.keys()))
                
                # Create header
                header = f"{'Model':<{model_width}} | {'Side':>5} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Range':>8}\n"
                separator = f"{'-' * model_width} | {'-' * 5} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8}\n"
                
                f.write(header)
                f.write(separator)
                
                # Write each row, sorted by model name
                for model in sorted(angle_stats.keys()):
                    stats = angle_stats[model]
                    # Write left knee stats
                    if stats['left']:
                        row = stats['left']
                        f.write(f"{model.upper():<{model_width}} | {'Left':>5} | {row['mean']:8.2f} | {row['std']:8.2f} | {row['min']:8.2f} | {row['max']:8.2f} | {row['range']:8.2f}\n")
                    # Write right knee stats
                    if stats['right']:
                        row = stats['right']
                        f.write(f"{model.upper():<{model_width}} | {'Right':>5} | {row['mean']:8.2f} | {row['std']:8.2f} | {row['min']:8.2f} | {row['max']:8.2f} | {row['range']:8.2f}\n")
            
            # Also save as CSV for further analysis
            # Prepare data for CSV
            csv_data = []
            for model in angle_stats:
                for side in ['left', 'right']:
                    if angle_stats[model][side]:
                        row = {'model': model.upper(), 'side': side}
                        row.update(angle_stats[model][side])
                        csv_data.append(row)
            
            # Convert to DataFrame and save
            stats_df = pd.DataFrame(csv_data)
            if not stats_df.empty:
                stats_df = stats_df.round(2)
                stats_df.to_csv(output_path / 'knee_angle_stats.csv', index=False)
                print("Knee angle statistics files generated successfully")
        else:
            print("\nNo knee angle statistics to report - no valid data found")

    def _output_outlier_stats(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Create a table showing detailed keypoint and movement statistics for each model."""
        stats = {}
        
        for model_name, df in data.items():
            model_key = model_name.lower()
            model_stats = {'left': {}, 'right': {}}
            
            for side in ['left', 'right']:
                # Get confidence scores
                knee_conf = df[f'{side}_knee_conf']
                hip_conf = df[f'{side}_hip_conf']
                ankle_conf = df[f'{side}_ankle_conf']
                foot_conf = df[f'{side}_foot_index_conf']
                
                # Calculate minimum confidence
                min_conf = min(
                    min(knee_conf) if not knee_conf.empty else 1.0,
                    min(hip_conf) if not hip_conf.empty else 1.0,
                    min(ankle_conf) if not ankle_conf.empty else 1.0,
                    min(foot_conf) if not foot_conf.empty else 1.0
                )
                
                # Calculate missing keypoint rate
                total_frames = len(df)
                missing_knee = sum((knee_conf < 0.5) | knee_conf.isna())
                missing_hip = sum((hip_conf < 0.5) | hip_conf.isna())
                missing_ankle = sum((ankle_conf < 0.5) | ankle_conf.isna())
                missing_foot = sum((foot_conf < 0.5) | foot_conf.isna())
                missing_rate = ((missing_knee + missing_hip + missing_ankle + missing_foot) / (4 * total_frames)) * 100
                
                # Calculate angle outliers
                angle_col = f'{side}_knee_angle'
                angles = pd.to_numeric(df[angle_col], errors='coerce')
                angle_mean = angles.mean()
                angle_std = angles.std()
                outlier_threshold = 2.0  # standard deviations
                outlier_mask = abs(angles - angle_mean) > (outlier_threshold * angle_std)
                outlier_count = sum(outlier_mask)
                outlier_rate = (outlier_count / total_frames) * 100
                
                # Calculate max deviation in standard deviations
                if outlier_count > 0:
                    deviations = abs(angles - angle_mean) / angle_std
                    max_deviation = max(deviations)
                else:
                    max_deviation = 0.0
                
                # Calculate average movements
                def calc_movement(x_col, y_col, z_col=None):
                    x_diff = df[x_col].diff().abs()
                    y_diff = df[y_col].diff().abs()
                    if z_col and z_col in df.columns:
                        z_diff = df[z_col].diff().abs()
                        movement = (x_diff**2 + y_diff**2 + z_diff**2).pow(0.5)
                    else:
                        movement = (x_diff**2 + y_diff**2).pow(0.5)
                    return movement.mean()
                
                # Calculate movement for each keypoint
                movements = [
                    calc_movement(f'{side}_hip_x', f'{side}_hip_y', f'{side}_hip_z'),
                    calc_movement(f'{side}_knee_x', f'{side}_knee_y', f'{side}_knee_z'),
                    calc_movement(f'{side}_ankle_x', f'{side}_ankle_y', f'{side}_ankle_z'),
                    calc_movement(f'{side}_foot_index_x', f'{side}_foot_index_y', f'{side}_foot_index_z')
                ]
                avg_movement = sum(movements) / len(movements)
                
                # Store statistics
                model_stats[side] = {
                    'knee': side,
                    'min_confidence': min_conf,
                    'missing_rate': missing_rate,
                    'outlier_count': outlier_count,
                    'outlier_rate': outlier_rate,
                    'max_deviation': max_deviation,
                    'avg_movement': avg_movement
                }
            
            stats[model_key] = model_stats
        
        if stats:
            # Create a formatted text file
            with open(output_path / 'keypoint_stats.txt', 'w') as f:
                f.write("Detailed Keypoint and Movement Statistics\n")
                f.write("=====================================\n\n")
                f.write("Note: Movement values are in pixels for 2D models and normalized coordinates for 3D models\n\n")
                
                # Calculate column widths
                model_width = max(len("Model"), max(len(model) for model in stats.keys()))
                
                # Create header
                header = (f"{'Model':<{model_width}} | {'Knee':>5} | {'Min Conf':>8} | {'Miss %':>7} | "
                         f"{'Out #':>5} | {'Out %':>6} | {'Max Dev':>7} | {'Avg Mov':>7}\n")
                separator = f"{'-' * model_width} | {'-' * 5} | {'-' * 8} | {'-' * 7} | {'-' * 5} | {'-' * 6} | {'-' * 7} | {'-' * 7}\n"
                
                f.write(header)
                f.write(separator)
                
                # Write each row, sorted by model name
                for model in sorted(stats.keys()):
                    model_stats = stats[model]
                    for side in ['left', 'right']:
                        row = model_stats[side]
                        f.write(f"{model.upper():<{model_width}} | {row['knee']:>5} | {row['min_confidence']:8.3f} | "
                               f"{row['missing_rate']:6.1f}% | {row['outlier_count']:5d} | {row['outlier_rate']:5.1f}% | "
                               f"{row['max_deviation']:7.1f} | {row['avg_movement']:7.2f}\n")
            
            # Also save as CSV for further analysis
            csv_data = []
            for model in stats:
                for side in ['left', 'right']:
                    row = {'model': model.upper()}
                    row.update(stats[model][side])
                    csv_data.append(row)
            
            # Convert to DataFrame and save
            stats_df = pd.DataFrame(csv_data)
            stats_df = stats_df.round(3)
            stats_df.to_csv(output_path / 'keypoint_stats.csv', index=False)

    def _plot_keypoint_movement(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Create scatter plots showing movement over time for each keypoint and leg."""
        keypoints = ['hip', 'knee', 'ankle', 'foot_index']
        sides = ['left', 'right']
        
        # Color scheme - consistent with inference speed plot
        colors = {
            'mediapipe 2d': '#2ca02c',    # green
            'mediapipe 3d': '#17becf',    # cyan
            'mmpose 2d': '#ff7f0e',       # orange
            'mmpose 3d': '#e377c2',       # pink
            'yolo 2d': '#8c564b'          # brown
        }
        
        # Window size for averaging movement (30 frames = 1 second at 30fps)
        window_size = 30
        
        # Create figure with subplots in 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each keypoint
        for idx, keypoint in enumerate(keypoints):
            ax = axes[idx]
            
            for model_name, df in data.items():
                model_key = model_name.lower()
                
                for side in sides:
                    # Calculate movement for each frame
                    x_col = f'{side}_{keypoint}_x'
                    y_col = f'{side}_{keypoint}_y'
                    z_col = f'{side}_{keypoint}_z'
                    
                    # Skip if columns don't exist
                    if x_col not in df.columns or y_col not in df.columns:
                        continue
                    
                    # Get the coordinates and handle missing values
                    x_vals = df[x_col]
                    y_vals = df[y_col]
                    
                    # Skip frames with missing values
                    valid_mask = ~(x_vals.isna() | y_vals.isna())
                    x_vals = x_vals[valid_mask]
                    y_vals = y_vals[valid_mask]
                    
                    # Calculate movement based on model type
                    if 'mediapipe' in model_key:
                        # MediaPipe coordinates are normalized [0,1]
                        # Scale to pixel space (assuming 1920x1080 as reference)
                        x_diff = x_vals.diff().abs() * 1920
                        y_diff = y_vals.diff().abs() * 1080
                    else:
                        # MMPose coordinates are already in pixel space
                        x_diff = x_vals.diff().abs()
                        y_diff = y_vals.diff().abs()
                    
                    # Calculate 2D movement by default
                    movement = (x_diff**2 + y_diff**2).pow(0.5)
                    
                    # Only include z-coordinate for 3D models and if z values are not None/NaN
                    if z_col in df.columns and not df[z_col].isna().all():
                        z_vals = df[z_col][valid_mask]
                        if 'mediapipe 3d' in model_key:
                            # MediaPipe 3D z-values are in meters. Keep them unscaled so movement is comparable.
                            z_diff = z_vals.diff().abs()
                        else:
                            z_diff = z_vals.diff().abs()
                        movement = (x_diff**2 + y_diff**2 + z_diff**2).pow(0.5)
                    
                    # Calculate average movement over windows, handling missing frames
                    frames = df['frame'][valid_mask].values
                    avg_movement = []
                    avg_frames = []
                    
                    for i in range(0, len(frames), window_size):
                        window_end = min(i + window_size, len(frames))
                        avg_movement.append(movement[i:window_end].mean())
                        avg_frames.append(frames[i:window_end].mean())
                    
                    # Plot averaged movement points with directional triangles
                    marker = '<' if side == 'left' else '>'  # Triangle pointing left/right
                    label = f"{model_name.upper()} ({side.title()})" if idx == 0 else None
                    ax.scatter(avg_frames, avg_movement,
                             label=label,
                             color=colors[model_key],
                             marker=marker,
                             s=150,  # Larger marker size for triangles
                             alpha=0.6)  # Some transparency
            
            # Customize subplot
            title = f'{keypoint.replace("_", " ").title()} Movement'
            ax.set_title(title)
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Movement Magnitude')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add overall title
        plt.suptitle(f'Keypoint Movement Analysis\nAveraged over {window_size} frames (1 second)', 
                    fontsize=16, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'keypoint_movement.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_visualizations(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Generate all visualizations and statistics."""
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate existing visualizations
        self._plot_inference_speed(data, output_dir)
        self._plot_model_comparison(data, output_dir)
        self._plot_angle_distributions(data, output_dir)
        self._plot_error_detection(data, output_dir)
        self._plot_confidence_timeline(data, output_dir)
        self._plot_stability_heatmaps(data, output_dir)
        self._plot_statistics_bars(data, output_dir)
        self._plot_keypoint_movement(data, output_dir)
        
        # Generate statistics files
        self._output_fps_stats(data, output_dir)
        self._output_angle_stats(data, output_dir)
        self._output_outlier_stats(data, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Compare knee angle tracking models')
    parser.add_argument('--output-dir', type=str, default='comparisons',
                      help='Directory to save comparison plots')
    parser.add_argument('--mp2d-csv', type=str,
                      help='CSV file from MediaPipe 2D model')
    parser.add_argument('--mp3d-csv', type=str,
                      help='CSV file from MediaPipe 3D model')
    parser.add_argument('--mmpose2d-csv', type=str,
                      help='CSV file from MMPose 2D model')
    parser.add_argument('--mmpose3d-csv', type=str,
                      help='CSV file from MMPose 3D model')
    parser.add_argument('--yolo2d-csv', type=str,
                      help='CSV file from YOLO 2D model')
    
    args = parser.parse_args()
    
    # Collect model files, skipping None values
    model_files = {
        'MediaPipe 2D': args.mp2d_csv,
        'MediaPipe 3D': args.mp3d_csv,
        'MMPose 2D': args.mmpose2d_csv,
        'MMPose 3D': args.mmpose3d_csv,
        'YOLO 2D': args.yolo2d_csv
    }
    
    # Filter out None values
    model_files = {k: v for k, v in model_files.items() if v is not None}
    
    if not model_files:
        print("Error: No model CSV files provided")
        parser.print_help()
        return
    
    # Initialize comparator and generate visualizations
    comparator = ModelComparator(model_files)
    data = comparator.load_data()
    
    if not data:
        print("Error: No valid data loaded from CSV files")
        return
    
    print("\nGenerating visualizations...")
    comparator.generate_visualizations(data, Path(args.output_dir))
    print(f"Visualizations saved in {args.output_dir}/")

if __name__ == "__main__":
    main() 