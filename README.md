# Disaster Damage Assessment using Deep Learning

## Overview
This project implements an automated disaster damage assessment pipeline using a 2D deep learning workflow. The system detects changes between pre-disaster and post-disaster satellite images and classifies building damage levels.

## Workflow
1. Image preprocessing (resize, alignment, normalization)
2. Change detection using Siamese U-Net
3. Building segmentation using YOLOv8-Seg
4. Damage assessment using Intersection over Area (IoA)

## Damage Classification
- Destroyed : > 70%
- Major Damage : 25–70%
- Minor Damage : 5–25%
- Safe : ≤ 5%

## Model Performance
Accuracy: 92.8%

## Files in this Repository
- model.py – Deep learning model implementation
- inference.py – Prediction pipeline
- app.py – Application script to run the system
- requirements.txt – Required Python libraries

## Installation
Install dependencies:

pip install -r requirements.txt

## Run the Project

python app.py

## Output
The system generates GIS-compatible damage assessment outputs.

## Future Work
The framework can be extended to include 3D reconstruction and advanced physical damage metrics when higher computational resources are available.

## Author
Ganadi Karthika Lakshmi Deepthi