# RWY.Vision — Runway Edge & Centerline Detection

A computer vision model that detects a runway's left edge, right edge, and centerline from a single image, trained on an 11GB dataset of runway imagery.

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

## Demo

<!-- Add a screenshot or GIF of the frontend here once you have one -->
<!-- ![demo](docs/demo.png) -->

## Overview

Given a single RGB image, the model predicts six anchor points defining three lines:
- **Left edge** (green)
- **Right edge** (red)
- **Centerline** (blue/cyan)

It uses a dual-head architecture on a shared MobileNetV3-Large backbone:
- A **segmentation head** predicting the runway surface mask
- A **regression head** predicting the six (x, y) anchor points directly

## Results

Evaluated on a held-out validation set of 733 images.

| Metric | Value |
|---|---|
| Mean IoU | 0.702 |
| Mean Line Error | 11.73 px |
| Epochs | 100 |
| LR schedule | Cosine decay |

## Project structure

```
.
├── inference.py          # Standalone inference: load checkpoint, predict on one image
├── app.py                 # FastAPI backend exposing POST /predict
├── runway_frontend.html   # Frontend demo UI (upload an image, see predicted lines)
├── runway_model_best.pth  # Trained model weights (not committed — see note below)
└── README.md
```

## Setup & running locally

```bash
pip install torch torchvision opencv-python matplotlib fastapi uvicorn python-multipart
```

**Option A — command line inference on a single image:**
```bash
python inference.py path/to/image.jpg
```

**Option B — run the web demo:**
```bash
# Terminal 1: start the backend
python app.py

# Terminal 2: serve the frontend
python -m http.server 8080
```
Then open `http://localhost:8080/runway_frontend.html` in your browser.

> **Note on the checkpoint:** `runway_model_best.pth` is not committed to this repo (see `.gitignore`) since GitHub isn't meant for large binary files. [Add a link here to where you're hosting it — Google Drive, Hugging Face, a GitHub Release, etc.]

## Training

- **Dataset:** ~11GB of runway imagery, 640×360 resolution
- **Backbone:** MobileNetV3-Large (ImageNet-pretrained, fine-tuned)
- **Loss:** Combined segmentation loss + anchor point regression loss
- **Training environment:** Google Colab (GPU)

## Known limitations

- **Domain gap:** the training data is sourced from flight-simulator imagery. The model performs well on validation data from the same distribution, but shows reduced accuracy on real-world aerial/ground photographs — particularly with asymmetric edge detection (one edge tracking more accurately than the other) on out-of-distribution camera angles. Closing this gap (via data augmentation or fine-tuning on real photos) is the main direction for future work.
- Performance on images shot from unusual angles or very low altitude has not been extensively validated.

## Tech stack

PyTorch · torchvision (MobileNetV3) · OpenCV · FastAPI · vanilla JS/HTML/CSS frontend
