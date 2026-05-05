import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_gradient_magnitude, uniform_filter
from sklearn.cluster import KMeans
from skimage.io import imsave
from skimage.util import img_as_ubyte


# ============================================================
# UNSUPERVISED SLIP-BAND IDENTIFICATION
# No labels are used.
# The model clusters pixels based on intensity, gradient, and texture.
# ============================================================

INPUT_MAP = "inputs_baseline/strain map.npy"
OUTPUT_DIR = "outputs_unsupervised"

N_CLUSTERS = 5
RANDOM_SEED = 42
DOWNSAMPLE = 1
EDGE_MARGIN = 25


def normalize_map(arr):
    arr = np.squeeze(arr).astype(np.float32)

    lo = np.percentile(arr, 1)
    hi = np.percentile(arr, 99)

    arr = (arr - lo) / (hi - lo + 1e-12)
    arr = np.clip(arr, 0, 1)

    return arr


def local_variance(img, size=9):
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img ** 2, size=size)
    var = mean_sq - mean ** 2
    return np.clip(var, 0, None)


def make_features(img):
    print("Extracting unsupervised features...")

    intensity = img

    gradient = gaussian_gradient_magnitude(img, sigma=1)
    gradient = normalize_map(gradient)

    texture = local_variance(img, size=9)
    texture = normalize_map(texture)

    features = np.stack(
        [
            intensity.ravel(),
            gradient.ravel(),
            texture.ravel()
        ],
        axis=1
    )

    return features, intensity, gradient, texture


def choose_slip_band_cluster(labels_img, intensity, gradient, texture):
    print("Choosing slip-band-like cluster...")

    best_cluster = None
    best_score = -1

    for cluster_id in range(N_CLUSTERS):
        mask = labels_img == cluster_id

        if np.sum(mask) == 0:
            continue

        mean_intensity = np.mean(intensity[mask])
        mean_gradient = np.mean(gradient[mask])
        mean_texture = np.mean(texture[mask])

        score = 0.4 * mean_intensity + 0.4 * mean_gradient + 0.2 * mean_texture

        print(
            f"Cluster {cluster_id}: "
            f"intensity={mean_intensity:.4f}, "
            f"gradient={mean_gradient:.4f}, "
            f"texture={mean_texture:.4f}, "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    print(f"Selected cluster: {best_cluster}")

    return (labels_img == best_cluster).astype(np.uint8), best_cluster


def save_cluster_image(labels_img, filename):
    plt.figure(figsize=(10, 10))
    plt.imshow(labels_img, cmap="tab10")
    plt.axis("off")
    plt.title("Unsupervised K-Means Clusters")
    plt.savefig(filename, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mask(mask, filename):
    imsave(filename, img_as_ubyte(mask.astype(np.float32)))


def save_coordinates(mask, filename):
    y, x = np.where(mask > 0)

    coords = pd.DataFrame({
        "x": x,
        "y": y
    })

    coords.to_csv(filename, index=False)

    return coords


def save_overlay(img, mask, filename):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img, cmap="gray", interpolation="nearest")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask == 1, 0] = 1.0
    overlay[mask == 1, 3] = 0.45

    ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")
    ax.set_title("Unsupervised Slip-Band Detection")

    plt.savefig(filename, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting unsupervised slip-band detection...")
    print("No labels are used.")

    strain_map = np.load(INPUT_MAP)
    strain_map = normalize_map(strain_map)
    strain_map[:EDGE_MARGIN, :] = np.median(strain_map)
    strain_map[-EDGE_MARGIN:, :] = np.median(strain_map)
    strain_map[:, :EDGE_MARGIN] = np.median(strain_map)
    strain_map[:, -EDGE_MARGIN:] = np.median(strain_map)

    print(f"Loaded strain map: {strain_map.shape}")

    if DOWNSAMPLE > 1:
        print(f"Downsampling by factor {DOWNSAMPLE}...")
        small_map = strain_map[::DOWNSAMPLE, ::DOWNSAMPLE]
    else:
        small_map = strain_map

    features, intensity, gradient, texture = make_features(small_map)

    print("Running K-Means clustering...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10
    )

    labels = kmeans.fit_predict(features)
    labels_img = labels.reshape(small_map.shape)

    slip_mask_small, chosen_cluster = choose_slip_band_cluster(
        labels_img,
        intensity,
        gradient,
        texture
    )

    if DOWNSAMPLE > 1:
        slip_mask = np.kron(
            slip_mask_small,
            np.ones((DOWNSAMPLE, DOWNSAMPLE), dtype=np.uint8)
        )

        labels_img_full = np.kron(
            labels_img,
            np.ones((DOWNSAMPLE, DOWNSAMPLE), dtype=np.uint8)
        )

        slip_mask = slip_mask[:strain_map.shape[0], :strain_map.shape[1]]
        slip_mask[:EDGE_MARGIN, :] = 0
        slip_mask[-EDGE_MARGIN:, :] = 0
        slip_mask[:, :EDGE_MARGIN] = 0
        slip_mask[:, -EDGE_MARGIN:] = 0
        labels_img_full = labels_img_full[:strain_map.shape[0], :strain_map.shape[1]]
    else:
        slip_mask = slip_mask_small
        labels_img_full = labels_img

    save_cluster_image(
        labels_img_full,
        os.path.join(OUTPUT_DIR, "unsupervised_clusters.png")
    )

    save_mask(
        slip_mask,
        os.path.join(OUTPUT_DIR, "unsupervised_slip_band_mask.png")
    )

    coords = save_coordinates(
        slip_mask,
        os.path.join(OUTPUT_DIR, "unsupervised_coordinates.csv")
    )

    save_overlay(
        strain_map,
        slip_mask,
        os.path.join(OUTPUT_DIR, "unsupervised_overlay.png")
    )

    print("DONE")
    print(f"Chosen cluster: {chosen_cluster}")
    print(f"Detected unsupervised slip-band pixels: {len(coords)}")

    print("Saved outputs:")
    print("- outputs_unsupervised/unsupervised_clusters.png")
    print("- outputs_unsupervised/unsupervised_slip_band_mask.png")
    print("- outputs_unsupervised/unsupervised_overlay.png")
    print("- outputs_unsupervised/unsupervised_coordinates.csv")


if __name__ == "__main__":
    main()