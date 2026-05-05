import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from torch.utils.data import DataLoader, Dataset, random_split

# This file implements a supervised ML prototype for slip-band pixel detection.
# It uses separate grain shear maps as inputs and the baseline mask as the label.
# The goal is to train a simple CNN to predict whether a pixel belongs to a slip band.

INPUT_DIR = "inputs_supervised/separate grain shear data"
BASELINE_MASK_PATH = "outputs_baseline/slip_band_mask.png"
BASELINE_FULL_MAP = "inputs_baseline/strain map.npy"
OUTPUT_DIR = "outputs_supervised"
PATCH_SIZE = 15
HALF_PATCH = PATCH_SIZE // 2
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
MAX_PATCHES_PER_CLASS_PER_MAP = 100


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_map(arr):
    arr = np.squeeze(arr).astype(np.float32)
    low = np.percentile(arr, 1)
    high = np.percentile(arr, 99)
    arr = (arr - low) / (high - low + 1e-12)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_baseline_label_mask(path):
    mask = imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() <= 1:
        mask = (mask > 0.5).astype(np.uint8)
    else:
        mask = (mask > 128).astype(np.uint8)
    return mask


def align_label_mask_to_sample(label_mask, target_shape):
    # Prototype label alignment: crop or pad the baseline label mask to the grain map shape.
    # In a later project stage, each separate grain map should have its own matching label mask.
    aligned = np.zeros(target_shape, dtype=np.uint8)
    min_rows = min(label_mask.shape[0], target_shape[0])
    min_cols = min(label_mask.shape[1], target_shape[1])
    aligned[:min_rows, :min_cols] = label_mask[:min_rows, :min_cols]
    return aligned


def load_grain_maps(input_dir):
    files = sorted(
        f
        for f in glob.glob(os.path.join(input_dir, "*.npy"))
        if not os.path.basename(f).startswith("._")
    )
    maps = []
    for path in files:
        arr = np.load(path)
        arr = normalize_map(arr)
        maps.append(arr)
    return maps


def extract_patch(sample, center_y, center_x, patch_size=PATCH_SIZE):
    half = patch_size // 2
    return sample[center_y - half : center_y + half + 1, center_x - half : center_x + half + 1]


def build_patch_dataset(grain_maps, label_mask):
    patches = []
    labels = []
    for sample in grain_maps:
        if sample.ndim != 2:
            continue
        target_label = align_label_mask_to_sample(label_mask, sample.shape)
        rows, cols = sample.shape
        if rows < PATCH_SIZE or cols < PATCH_SIZE:
            continue
        valid_centers = np.zeros_like(target_label, dtype=bool)
        valid_centers[HALF_PATCH : rows - HALF_PATCH, HALF_PATCH : cols - HALF_PATCH] = True
        pos_centers = np.argwhere(valid_centers & (target_label == 1))
        neg_centers = np.argwhere(valid_centers & (target_label == 0))
        if len(pos_centers) == 0 or len(neg_centers) == 0:
            continue
        num = min(len(pos_centers), len(neg_centers), MAX_PATCHES_PER_CLASS_PER_MAP)
        if len(pos_centers) > num:
            pos_centers = pos_centers[np.random.choice(len(pos_centers), num, replace=False)]
        if len(neg_centers) > num:
            neg_centers = neg_centers[np.random.choice(len(neg_centers), num, replace=False)]
        for y, x in pos_centers:
            patch = extract_patch(sample, y, x)
            patches.append(patch)
            labels.append(1)
        for y, x in neg_centers:
            patch = extract_patch(sample, y, x)
            patches.append(patch)
            labels.append(0)
    if len(patches) == 0:
        raise RuntimeError("No training patches were created. Check input data and baseline label mask.")
    patches = np.stack(patches, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.float32)
    return patches, labels


class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.from_numpy(patches).unsqueeze(1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


class SlipBandCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(1)


def train_model(model, train_loader, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions.long() == targets.long()).sum().item()
                total += targets.size(0)
        val_accuracy = correct / max(total, 1)
        print(f"Epoch {epoch}/{EPOCHS} - train loss: {train_loss:.4f} - val accuracy: {val_accuracy:.4f}")
    return model


def save_prediction_mask(pred_mask, filename):
    imsave(filename, img_as_ubyte(pred_mask.astype(np.uint8)))


def save_probability_image(prob_map, filename):
    imsave(filename, img_as_ubyte(np.clip(prob_map, 0.0, 1.0)))


def save_predicted_coordinates(pred_mask, filename):
    y, x = np.where(pred_mask > 0)
    coords = pd.DataFrame({"x": x, "y": y})
    coords.to_csv(filename, index=False)
    return coords


def save_supervised_overlay(full_map, pred_mask, filename):
    norm_map = normalize_map(full_map)
    overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    overlay[pred_mask == 1, 0] = 1.0
    overlay[pred_mask == 1, 3] = 0.4
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(norm_map, cmap="gray", interpolation="nearest")
    ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")
    plt.savefig(filename, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def predict_full_map(model, full_map, device):
    model.eval()
    h, w = full_map.shape
    padded = np.pad(full_map, HALF_PATCH, mode="reflect")
    padded_tensor = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    prob_map = np.zeros((h, w), dtype=np.float32)
    row_chunk = 8
    for row in range(0, h, row_chunk):
        row_end = min(row + row_chunk, h)
        chunk = padded_tensor[:, :, row : row_end + PATCH_SIZE - 1, :]
        patches = F.unfold(chunk, kernel_size=PATCH_SIZE).transpose(1, 2)
        all_probs = []
        with torch.no_grad():
            for start in range(0, patches.size(0), BATCH_SIZE):
                batch = patches[start : start + BATCH_SIZE]
                batch = batch.view(-1, 1, PATCH_SIZE, PATCH_SIZE).to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        all_probs = np.concatenate(all_probs, axis=0)
        prob_map[row:row_end, :] = all_probs.reshape(row_end - row, w)
    return prob_map


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_random_seed(SEED)

    print("Loading supervised training data...")
    grain_maps = load_grain_maps(INPUT_DIR)
    label_mask = load_baseline_label_mask(BASELINE_MASK_PATH)
    patches, labels = build_patch_dataset(grain_maps, label_mask)

    dataset = PatchDataset(patches, labels)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = SlipBandCNN().to(device)

    print("Training supervised slip-band classifier...")
    model = train_model(model, train_loader, val_loader, device)

    model_path = os.path.join(OUTPUT_DIR, "supervised_slip_band_model.pth")
    torch.save(model.state_dict(), model_path)

    print("Predicting on the full baseline strain map...")
    full_map = normalize_map(np.load(BASELINE_FULL_MAP))
    prob_map = predict_full_map(model, full_map, device)
    pred_mask = (prob_map >= 0.5).astype(np.uint8)

    save_probability_image(prob_map, os.path.join(OUTPUT_DIR, "predicted_slip_band_probability.png"))
    save_prediction_mask(pred_mask, os.path.join(OUTPUT_DIR, "predicted_slip_band_mask.png"))
    predicted_coords = save_predicted_coordinates(pred_mask, os.path.join(OUTPUT_DIR, "predicted_slip_band_coordinates.csv"))
    save_supervised_overlay(full_map, pred_mask, os.path.join(OUTPUT_DIR, "supervised_overlay.png"))

    print("Saved supervised outputs:")
    print(f"- {model_path}")
    print(f"- {os.path.join(OUTPUT_DIR, 'predicted_slip_band_probability.png')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'predicted_slip_band_mask.png')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'predicted_slip_band_coordinates.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'supervised_overlay.png')}")
    print(f"Training and prediction completed with {len(predicted_coords)} predicted slip-band pixels.")


if __name__ == "__main__":
    main()
