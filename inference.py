"""
Runway Line Detection — Inference Script
==========================================
This loads your trained model and runs it on a single image to predict
runway lines (left edge, right edge, centerline).

HOW TO USE:
    python inference.py path/to/your/image.jpg

WHAT YOU NEED:
    1. runway_model_best.pth  (your downloaded checkpoint — same folder as this script,
       or update CHECKPOINT_PATH below)
    2. pip install torch torchvision opencv-python matplotlib
"""

import sys
import torch
import torch.nn as nn
from torchvision import models
import cv2
import matplotlib.pyplot as plt

# -----------------------------
# 1. SAME MODEL CLASS AS TRAINING
# -----------------------------
# This has to match your Colab model EXACTLY — it's the "blueprint" that
# the saved numbers (.pth file) get loaded into.
class RunwayNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)  # weights=None: we're loading OUR trained weights, not ImageNet's
        self.features = backbone.features

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(960, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

        self.regressor = nn.Sequential(
            nn.Conv2d(960, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        feats = self.features(x)
        seg = self.decoder(feats).squeeze(1)
        anchors = self.regressor(feats)
        return seg, anchors


# -----------------------------
# 2. SETTINGS — edit if needed
# -----------------------------
CHECKPOINT_PATH = "runway_model_best.pth"   # path to your downloaded checkpoint
IMG_SIZE = 256                              # model was trained on 256x256 input
TRAIN_ASPECT = 640 / 360                    # aspect ratio of the images the model was trained on
PAD_COLOR = (114, 114, 114)                 # neutral grey for the padding bars
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 3. LOAD MODEL (do this once)
# -----------------------------
def load_model():
    model = RunwayNet().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()  # inference mode: turns off dropout etc.
    return model


# -----------------------------
# 3b. FIX ASPECT RATIO BEFORE RESIZING
# -----------------------------
# Training images were always 640x360 (16:9), stretched directly into a
# 256x256 square. A photo with a DIFFERENT aspect ratio gets squished by a
# different amount than anything the model has seen, which distorts the
# runway's geometry and throws predictions off.
#
# Fix: pad the image with neutral grey bars so its aspect ratio becomes
# 16:9 BEFORE the resize step. That way the final stretch-to-square always
# behaves the way it did during training, no matter what shape the input is.
def pad_to_train_aspect(img):
    h, w = img.shape[:2]
    current_aspect = w / h

    if abs(current_aspect - TRAIN_ASPECT) < 1e-3:
        return img, 0, 0  # already the right shape, nothing to do

    if current_aspect < TRAIN_ASPECT:
        # image is too narrow/tall relative to 16:9 -> add bars on left/right
        new_w = int(round(h * TRAIN_ASPECT))
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=PAD_COLOR)
        return padded, pad_left, 0
    else:
        # image is too wide relative to 16:9 -> add bars on top/bottom
        new_h = int(round(w / TRAIN_ASPECT))
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=PAD_COLOR)
        return padded, 0, pad_top


# -----------------------------
# 4. RUN ON ONE IMAGE
# -----------------------------
def predict(model, image_path):
    # Read image and remember its original size (so we can scale predictions back)
    orig = cv2.imread(image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Pad to match the training aspect ratio, THEN resize/stretch to square —
    # this matches exactly what every training image went through.
    padded, pad_left, pad_top = pad_to_train_aspect(orig)
    ph, pw = padded.shape[:2]

    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)  # add batch dimension: [1, 3, 256, 256]

    with torch.no_grad():
        seg_out, anchor_out = model(tensor)

    # anchor_out is 12 numbers in [0,1]: 6 points (x,y) = left edge (2 pts),
    # right edge (2 pts), centerline (2 pts) — normalized to the PADDED image
    anchors = anchor_out.squeeze(0).cpu().numpy()
    anchors[0::2] *= pw   # scale x to padded-canvas pixel coords
    anchors[1::2] *= ph   # scale y to padded-canvas pixel coords

    # shift back to the ORIGINAL image's coordinate frame by removing the padding offset
    anchors[0::2] -= pad_left
    anchors[1::2] -= pad_top

    return orig, anchors


# -----------------------------
# 5. DRAW THE PREDICTED LINES
# -----------------------------
def draw_lines(image, anchors):
    img = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # green=left, red=right, blue=center
    labels = ["Left edge", "Right edge", "Centerline"]

    for i in range(3):
        x1, y1, x2, y2 = anchors[i*4 : i*4+4]
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(img, pt1, pt2, colors[i], 3, cv2.LINE_AA)

    return img, labels


# -----------------------------
# 6. MAIN
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading model...")
    model = load_model()

    print(f"Running prediction on {image_path}...")
    orig_image, anchors = predict(model, image_path)
    result_image, labels = draw_lines(orig_image, anchors)

    plt.figure(figsize=(10, 6))
    plt.imshow(result_image)
    plt.title("Predicted Runway Lines (Green=Left, Red=Right, Blue=Center)")
    plt.axis("off")
    plt.savefig("prediction_output.png", bbox_inches="tight")
    print("Saved result to prediction_output.png")
    plt.show()