# Augmenting Knee Rehabilitation Replays Using MediaPipe Pose Estimation

## Overview
This project implements a methodology for analysing knee angles from video footage using MediaPipe Pose estimation. It provides near real-time visualisation $${\large\color{dodgerblue}&#x24D8; }$$  and analysis of knee movements, supporting both front-view and side-view perspectives.  

> [!NOTE]
> Near real-time visualisation of MP4 videos depends on the computer specification. <br/>
> <ins>Hint</ins>: If using a decade old laptop for fast scrolling and A-B sequence analysis, consider capturing first a screen cast of the generated augmented video replay.
> 
> As a stand-alone application, this source code is the second part of the three stage processing workflow, which is also a part of it's parent project codebase [^1] intended for home use and advancements of near-future analytical healthcare systems. See more: "*Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study*" [^1][^2].
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
If running the project on a computer with Python and OpenCV installed, install Google MediaPipe by:
```bash
pip install mediapipe
```
or by preparing your own `requirements.txt` for project deployments on multiple machines of similar specifications:
```bash
pip install -r requirements.txt
```
## Usage
To run this application, at `Windows` Command Prompt, or in `MacOS`/`Linux` Terminal, use the following sytax:
``` 
python main.py video_file output_csv [--export_knee {left,right,both}] [--direction {left,right,forward}]

Arguments
video_file: Path to input video file
output_csv: Path for output CSV file
--export_knee: Specify which knee to analyze (default: both)
--direction: Override foot direction detection
```
### Citation
If using our code or models for your research, please cite [^1] or use BibTeX format:
```
@inproceedings{bbacic2024simple,
    author={Bačić, Boris and Vasile, Claudiu and Feng, Chengwei and Ciucă, Marian},
    title={Towards nation-wide analytical healthcare infrastructures: A privacy-preserving augmented knee rehabilitation case study},
    booktitle = {Conference on Innovative Technologies in Intelligent Systems & Industrial Applications (CITISIA 2024)},
    year = {2024}
}
```
If using **LaTeX** or **Overleaf**, here is a recommended BibTeX format (to preserve special characters):


&nbsp;
### Contributors
| Contributor | Description | 
| :--- | --- |
|**Claudiu Vasile** [@claudiunz](https://github.com/claudiunz) | Python code, incremental prototyping, testing, and development. <br/> Github main project [tnwahi-appakrcs](https://github.com/claudiunz/tnwahi-appakrcs) |
|**Dr Marian G. Ciucă** &nbsp; | Computational geometry and trigonometric equations for knee angle calculations. |
|**Chengwei Feng** | Dataset labelling verification, models and code templates alternatives. | 
|**Dr Boris Bačić**  <br/> [@bbacic](https://github.com/bbacic) | Team leader, supervision, project code reviewing, and project codebase[^1] development, testing and integration.  GitHub project reviewer and co-contributor<sup>1)</sup>. <br/> | 

<sup>1)</sup>Note: https://github.com/bbacic/tnwahi-appakrcs/ is a contributing fork of [tnwahi-appakrcs](https://github.com/claudiunz/tnwahi-appakrcs). 
<br/>

### Version History of the [tnwahi-appakrcs](https://github.com/claudiunz/tnwahi-appakrcs) Project
| Version | Date | Summary/Action/Rationale/Acknowledgements | Project/Filename | 
| :--- | --- |  --- | --- |
| 3.2 | Dec 2024 | CITISIA 2024 publication [^1]. <br/> Refinements, issues and bug fixes of PoC (ver. 3) for the scope of the publication.| main.py |
| 3 | Nov 2024 | Front and side camera view with extended diagnostic information processing (from ver. 2 PoC) | main.py <- script.py |
| 2 | May 2024 | Privacy-preserving augmented video analysis with diagnostic information streaming and visualisation (PoC)| script.py |
| 1 | Aug 2023 | Initial project scoping: Specifications and development options for knee exercise. <br/> Beta version of Prototyping and proof-of-concept (PoC). | script.py   |

&nbsp;
### Reference
