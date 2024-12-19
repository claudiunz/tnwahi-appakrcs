# Towards Nation-wide Analytical Healthcare Infrastructures: A Privacy-Preserving Augmented Knee Rehabilitation Case Study

## Overview
This project implements a methodology for analyzing knee angles from video footage using MediaPipe Pose estimation. It provides real-time visualization and analysis of knee movements, supporting both front-view and side-view perspectives.

## Features
- Real-time knee angle detection and measurement
- Support for both left and right knee analysis
- Front-view and side-view analysis capabilities
- CSV export of tracking data
- Real-time visualization with angle plots
- Incorrect movement detection

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- Matplotlib
- NumPy

## Installation
```bash
pip install -r requirements.txt

## Usage
python main.py video_file output_csv [--export_knee {left,right,both}] [--direction {left,right,forward}]

Arguments
video_file: Path to input video file
output_csv: Path for output CSV file
--export_knee: Specify which knee to analyze (default: both)
--direction: Override foot direction detection
