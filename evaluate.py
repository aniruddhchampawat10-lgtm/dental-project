import torch
import csv
import os
import numpy as np

from dataset import DentalDataset
from model import get_model
from utils import *

# -------------------------------------------------
# YOUR EXACT PATHS
# -------------------------------------------------
IMAGE_DIR = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\data\images"
MASK_DIR  = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\data\masks"
MODEL_PATH = r"C:\Users\aniru\models\best_model.pth"




OUTPUT_DIR = r"C:\Users\aniru\OneDrive\Desktop\Dental_Caries_Segmentation\outputs"
# -------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Safety checks (important)
assert os.path.exists(IMAGE_DIR), f"Images folder not found: {IMAGE_DIR}"
assert os.path.exists(MASK_DIR), f"Masks folder not found: {MASK_DIR}"
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

# Load dataset
dataset = DentalDataset(IMAGE_DIR, MASK_DIR)

# Load model
model = get_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

for img, mask, name in dataset:
    img = img.unsqueeze(0).to(device)
    mask = mask.numpy()

    with torch.no_grad():
        pred = torch.sigmoid(model(img)).cpu().numpy()
        pred = (pred > 0.5).astype("float32")

    d = dice(pred, mask)
    j = iou(pred, mask)
    p = precision(pred, mask)
    r = recall(pred, mask)
    f1 = 2 * p * r / (p + r + 1e-6)
    acc = (pred == mask).mean()
    h = hausdorff(pred[0, 0], mask[0, 0])

    rows.append([name, d, j, p, r, f1, acc, h])

# Save metrics
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")

with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["Image", "Dice", "IoU", "Precision", "Recall", "F1", "Accuracy", "Hausdorff"]
    )
    writer.writerows(rows)

print("Evaluation completed successfully.")
print("Metrics saved to:", metrics_path)
