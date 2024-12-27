# Augmenting Knee Rehabilitation Replays Using MediaPipe Pose Estimation

## Overview
This project implements a methodology for analyzing knee angles from video footage using MediaPipe Pose estimation. It provides near real-time visualization (see the Note below) and analysis of knee movements, supporting both front-view and side-view perspectives.  

> [!NOTE]
> Near real-time visualisation of MP4 videos depends on the computer.
> <ins>Hint</ins>: If using a decade old laptop for fast scrolling and A-B sequence analysis, consider capturing first a screen cast of the augmented video replay.
> 
> This source code is the second part of the three stage processing workflow reported in privacy-preserving augmented knee rehabilitation case study intended for home use and near-future analytical healthcare systems [^1][^2]
>

[^1]: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). *Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study*. Presented at the meeting of the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW. [In print].
[^2]: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). _Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study_. Presented at the meeting of the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW. [ArXiv preprint].

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
```
## Usage
``` 
python main.py video_file output_csv [--export_knee {left,right,both}] [--direction {left,right,forward}]

Arguments
video_file: Path to input video file
output_csv: Path for output CSV file
--export_knee: Specify which knee to analyze (default: both)
--direction: Override foot direction detection
```
### Contributors
Python code prototyping, testing, and development: Claudiu Vasile

GitHub project shared by: Claudiu Vasile

Computational geometry and trigonometric equations: Dr Marian G. Ciucă

Dataset labelling verification, models and code templates alternatives: Chengwei Feng

Project leader, supervision, project code testing, privacy-preserving augmented near-real time replay with summary concept for Privacy-Preserving Augmented Knee Rehabilitation Case Study for Home Use and Analytical Healthcare Systems: Dr Boris Bačić

Integration of CSV data streaming verification, adaptive exercise indexing, visualisation and summary (implemented separately in Matlab) as reported in the paper [^1] (and it's pre-print [^2]): Dr Boris Bačić

Knee rehabilitation dataset recording protocol, video capture, conversion and labelling: Dr Boris Bačić 

GitHub project reviewer and co-contributor: Dr Boris Bačić

### Reference
