import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import DentalDataset
from model import get_model

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "../data/images"
MASK_DIR = "../data/masks"
MODEL_PATH = "../outputs/models/best_model.pth"
SAVE_LIMIT = 20        # number of samples to save
THRESHOLD = 0.5
# ----------------------------------------

# Create output directories
os.makedirs("../outputs/predictions", exist_ok=True)
os.makedirs("../outputs/overlays", exist_ok=True)
os.makedirs("../outputs/error_maps", exist_ok=True)
os.makedirs("../outputs/case_studies", exist_ok=True)

# Load dataset
dataset = DentalDataset(IMAGE_DIR, MASK_DIR)

# Load model
model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model and dataset loaded. Generating visualizations...")

for idx in range(min(SAVE_LIMIT, len(dataset))):
    image, mask, name = dataset[idx]

    image = image.to(DEVICE)
    mask_np = mask[0].numpy()

    # ---------------- PREDICTION ----------------
    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0)))
        pred = (pred.cpu().numpy()[0, 0] > THRESHOLD).astype("uint8")

    # ---------------- SAVE PREDICTION ----------------
    cv2.imwrite(
        f"../outputs/predictions/{name}",
        pred * 255
    )

    # ---------------- PREPARE IMAGES ----------------
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_gray = (img_np[:, :, 0] * 255).astype("uint8")

    # ---------------- OVERLAY ----------------
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay[pred == 1] = [0, 0, 255]  # Red = predicted lesion

    cv2.imwrite(
        f"../outputs/overlays/{name}",
        overlay
    )

    # ---------------- ERROR MAP ----------------
    error_map = np.zeros((256, 256, 3), dtype=np.uint8)

    # False Positive (Red)
    error_map[(pred == 1) & (mask_np == 0)] = [255, 0, 0]

    # False Negative (Blue)
    error_map[(pred == 0) & (mask_np == 1)] = [0, 0, 255]

    cv2.imwrite(
        f"../outputs/error_maps/{name}",
        error_map
    )

    # ---------------- SIDE-BY-SIDE + CASE STUDY ----------------
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].imshow(img_gray, cmap="gray")
    axs[0].set_title("Original X-ray")

    axs[1].imshow(mask_np, cmap="gray")
    axs[1].set_title("Ground Truth Mask")

    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Predicted Mask")

    axs[3].imshow(overlay)
    axs[3].set_title("Overlay")

    axs[4].imshow(error_map)
    axs[4].set_title("Error Map")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"../outputs/case_studies/{name}")
    plt.close()

print("🎉 Visualization completed successfully!")
