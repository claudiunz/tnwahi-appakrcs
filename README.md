# Augmenting Knee Rehabilitation Replays Using MediaPipe Pose Estimation

## Overview
This project implements a methodology for analysing knee angles from video footage using MediaPipe Pose estimation. It provides near real-time visualisation $$\text{\color{dodgerblue} &#x24D8;}$$ and analysis of knee movements, supporting both front-view and side-view perspectives.  

> [!NOTE]
> Near real-time visualisation of MP4 videos depends on the computer.
> <ins>Tip</ins>: If using a decade old laptop for fast scrolling and A-B sequence analysis, consider capturing first a screen cast of the augmented video replay.
> 
> As a stand-alone project, this source code is the second part of the three stage processing workflow intended for home use and near-future analytical healthcare systems. See more: Privacy-preserving augmented knee rehabilitation case study [^1][^2].
>

[^1]: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). *Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study*. Presented at the meeting of the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW. [In print].
[^2]: Bačić, B., Claudiu Vasile, Feng, C., & Ciucă, M. G. (2024, 13-15 Dec.). _Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study_. Presented at the meeting of the Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024), Sydney, NSW. [ArXiv preprint].

## Features
- Near real-time knee angle detection and measurement
- Support for both left and right knee analysis
- Front-view and side-view analysis capabilities
- CSV export of tracking data
- Real-time visualization with angle plots
- Incorrect knee movement detection

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
| Contributor | Description | 
| :--- | --- |
|**Claudiu Vasile** [@claudiunz](https://github.com/claudiunz) | Python code prototyping, testing, and development. <br/> Github main project [tnwahi-appakrcs](https://github.com/claudiunz/tnwahi-appakrcs) |
|**Dr Marian G. Ciucă** | Computational geometry and trigonometric equations for knee agle calculations. |
|**Chengwei Feng** | Dataset labelling verification, models and code templates alternatives. | 
|**Dr Boris Bačić** [@bbacic](https://github.com/bbacic) | *Privacy-Preserving Augmented Knee Rehabilitation Case Study for Home Use and Analytical Healthcare Systems* concept inventor and project leader. <br/> Supervision, project code reviews, bug and issues reporting, and overall project testing. <br/> Privacy-preserving augmented near-real time replay with extracted diagnostic information timeseries streaming and summary concept applied on *Privacy-Preserving Augmented Knee Rehabilitation Case Study for Home Use and Analytical Healthcare Systems*. <br/> CSV data streaming integration and verification. <br/> Author and developer of adaptive and unsupervised approach for knee exercise recognition and timeline indexing, visualisation and summary (implemented separately in Matlab) as the third  stage processing workflow [^1] (see also the pre-print [^2] of [^1]). <br/> Knee rehabilitation dataset recording protocol. <br/> Case study video capture, files conversion and labelling. <br/> GitHub project reviewer and co-contributor. Note: https://github.com/bbacic/tnwahi-appakrcs/ is a contributing fork of [tnwahi-appakrcs](https://github.com/claudiunz/tnwahi-appakrcs). <br/> | 


### Reference
