import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, convolve
from skimage.draw import line
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.io import imsave
from skimage.util import img_as_ubyte

# Baseline classical image-processing pipeline for slip band detection.
# This code is intentionally not a machine learning model yet.
# The output can be used later as labels for ML pixel-wise classification/segmentation.

# ---------------- PARAMETERS ----------------

FNAME = "strain map.npy"
SIGMA_BLUR = 1.0
EDGE_MARGIN = 10
LINE_SCALES = [15, 25]
THETAS = np.arange(0, 180, 20)
THRESHOLD_PERCENTILE = 96
MIN_NEIGHBOURS = 2
TILE_SIZE = 40
MIN_PIXELS_PER_TILE = 12

# Output directory for baseline method
OUTPUT_DIR = "outputs_baseline"

# ---------------- HELPERS ----------------

def mat2gray_manual(img):
    mn = np.min(img)
    mx = np.max(img)
    return np.clip((img - mn) / (mx - mn + 1e-12), 0, 1)


def save_img(img, filename):
    imsave(filename, img_as_ubyte(mat2gray_manual(img)))


def save_colormap_image(img, filename, cmap="inferno", dpi=600, colorbar=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mat2gray_manual(img), cmap=cmap, interpolation="nearest")
    ax.axis("off")
    if colorbar:
        fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.02)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def enhance_visual_contrast(img, clip_percentiles=(2, 98), clip_limit=0.03):
    low, high = np.percentile(img[~np.isnan(img)], clip_percentiles)
    img_clipped = np.clip(img, low, high)
    img_rescaled = rescale_intensity(img_clipped, in_range=(low, high), out_range=(0, 1))
    return equalize_adapthist(img_rescaled, clip_limit=clip_limit)


def save_overlay_slip_bands(I, BW, filename, cmap="inferno", dpi=600):
    I_vis = enhance_visual_contrast(I)
    overlay = np.zeros((*BW.shape, 4), dtype=float)
    overlay[BW, 0] = 1.0
    overlay[BW, 3] = 0.45
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(I_vis, cmap=cmap, interpolation="nearest")
    ax.imshow(overlay, interpolation="nearest")
    ax.axis("off")
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ---------------- PIPELINE FUNCTIONS ----------------

def load_strain_map(fname):
    E = np.load(fname)
    return np.squeeze(E).astype(float)


def create_edge_mask(shape, edge_margin):
    mask = np.ones(shape, dtype=bool)
    mask[:edge_margin, :] = False
    mask[-edge_margin:, :] = False
    mask[:, :edge_margin] = False
    mask[:, -edge_margin:] = False
    return mask


def normalise_strain_map(E, mask):
    valid_E = E[mask]
    lo = np.percentile(valid_E, 1)
    hi = np.percentile(valid_E, 99)
    I = np.clip((E - lo) / (hi - lo + 1e-12), 0, 1)
    return I


def high_pass_enhancement(I, mask, sigma_blur):
    bg = np.median(I[mask])
    I_masked = I.copy()
    I_masked[~mask] = bg
    I_s = gaussian_filter(I_masked, sigma=sigma_blur)
    I_hp = I_s - gaussian_filter(I_s, sigma=4)
    I_hp = mat2gray_manual(I_hp)
    I_hp[~mask] = 0
    return I_hp


def make_line_kernel(L, theta_deg):
    kernel = np.zeros((L, L), dtype=float)
    c = L // 2
    theta = np.deg2rad(theta_deg)
    x0 = int(round(c - ((L - 1) / 2) * np.cos(theta)))
    y0 = int(round(c - ((L - 1) / 2) * np.sin(theta)))
    x1 = int(round(c + ((L - 1) / 2) * np.cos(theta)))
    y1 = int(round(c + ((L - 1) / 2) * np.sin(theta)))
    rr, cc = line(y0, x0, y1, x1)
    valid = (rr >= 0) & (rr < L) & (cc >= 0) & (cc < L)
    kernel[rr[valid], cc[valid]] = 1
    return kernel / (kernel.sum() + 1e-12)


def detect_slip_band_response(I_hp, mask, line_scales, thetas):
    kernels = []
    for L in line_scales:
        for th in thetas:
            kernels.append(make_line_kernel(L, th))

    resp = np.zeros_like(I_hp)
    total = len(kernels)
    for count, k in enumerate(kernels, 1):
        print(f"Processing kernel {count}/{total}")
        r = convolve(I_hp, k, mode="nearest")
        resp = np.maximum(resp, r)

    resp = mat2gray_manual(resp)
    resp[~mask] = 0
    return resp


def threshold_response(resp, mask, threshold_percentile):
    valid_resp = resp[mask]
    p_thresh = np.percentile(valid_resp, threshold_percentile)
    BW = resp > p_thresh
    BW[~mask] = False
    return BW


def remove_noise(BW, mask, min_neighbours):
    neighbour_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_count = convolve(BW.astype(np.uint8), neighbour_kernel, mode="constant", cval=0)
    BW_clean = BW & (neighbour_count >= min_neighbours)
    BW_clean[~mask] = False
    return BW_clean


def save_dataframe_to_csv(df, filename):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except PermissionError:
            pass
    df.to_csv(filename, index=False)


def save_pixel_coordinates(BW, filename):
    y, x = np.where(BW)
    pixel_coords = pd.DataFrame({"x": x, "y": y})
    save_dataframe_to_csv(pixel_coords, filename)
    return pixel_coords


def locate_slip_band_regions(BW, I, tile_size, min_pixels_per_tile, image_filename, csv_filename):
    circle_data = []
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(I, cmap="inferno", interpolation="nearest")
    ax.axis("off")
    ax.set_title("Located Slip Bands", color="white", pad=16)

    rows, cols = BW.shape
    for r0 in range(0, rows, tile_size):
        for c0 in range(0, cols, tile_size):
            r1 = min(r0 + tile_size, rows)
            c1 = min(c0 + tile_size, cols)
            tile = BW[r0:r1, c0:c1]
            ty, tx = np.where(tile)
            if len(tx) >= min_pixels_per_tile:
                global_x = tx + c0
                global_y = ty + r0
                cx = np.mean(global_x)
                cy = np.mean(global_y)
                radius = max(
                    np.max(np.abs(global_x - cx)),
                    np.max(np.abs(global_y - cy)),
                    8,
                )
                circle = plt.Circle((cx, cy), radius, color="yellow", fill=False, linewidth=3)
                ax.add_patch(circle)
                ax.text(
                    cx,
                    cy,
                    f"({cx:.0f},{cy:.0f})",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "yellow", "alpha": 0.7, "edgecolor": "black", "boxstyle": "round,pad=0.2"},
                )
                circle_data.append([cx, cy, radius, len(tx)])

    fig.savefig(image_filename, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    circle_df = pd.DataFrame(circle_data, columns=["cx", "cy", "radius", "pixel_count"])
    save_dataframe_to_csv(circle_df, csv_filename)
    return circle_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting fast slip-band detection...")
    E = load_strain_map(FNAME)
    rows, cols = E.shape
    print(f"Loaded strain map: {rows} x {cols}")

    mask = create_edge_mask(E.shape, EDGE_MARGIN)
    I = normalise_strain_map(E, mask)

    I_vis = enhance_visual_contrast(I)
    save_colormap_image(I_vis, os.path.join(OUTPUT_DIR, "normalised_strain_map.png"), cmap="inferno", dpi=600)

    I_hp = high_pass_enhancement(I, mask, SIGMA_BLUR)
    I_hp_vis = enhance_visual_contrast(I_hp)
    save_colormap_image(I_hp_vis, os.path.join(OUTPUT_DIR, "high_pass_enhanced.png"), cmap="inferno", dpi=600)

    print("Detecting directional slip-band response...")
    resp = detect_slip_band_response(I_hp, mask, LINE_SCALES, THETAS)
    save_colormap_image(resp, os.path.join(OUTPUT_DIR, "slip_band_response.png"), cmap="inferno", dpi=600)

    print("Thresholding slip-band pixels...")
    BW = threshold_response(resp, mask, THRESHOLD_PERCENTILE)

    print("Removing noise...")
    BW = remove_noise(BW, mask, MIN_NEIGHBOURS)
    imsave(os.path.join(OUTPUT_DIR, "slip_band_mask.png"), img_as_ubyte(BW.astype(float)))

    pixel_coords = save_pixel_coordinates(BW, os.path.join(OUTPUT_DIR, "slip_band_pixel_coordinates.csv"))

    save_overlay_slip_bands(I, BW, os.path.join(OUTPUT_DIR, "overlay_slip_bands.png"))

    print("Locating slip-band regions...")
    circle_df = locate_slip_band_regions(
        BW,
        I,
        TILE_SIZE,
        MIN_PIXELS_PER_TILE,
        os.path.join(OUTPUT_DIR, "located_slip_bands.png"),
        os.path.join(OUTPUT_DIR, "slip_band_locations.csv"),
    )

    print("DONE")
    print(f"Detected slip-band pixels: {len(pixel_coords)}")
    print(f"Located slip-band regions: {len(circle_df)}")
    print("Saved files:")
    print("- normalised_strain_map.png")
    print("- high_pass_enhanced.png")
    print("- slip_band_response.png")
    print("- slip_band_mask.png")
    print("- overlay_slip_bands.png")
    print("- located_slip_bands.png")
    print("- slip_band_pixel_coordinates.csv")
    print("- slip_band_locations.csv")
    print("Baseline complete. Next step: use this output as labels for ML pixel-wise classification/segmentation.")


if __name__ == "__main__":
    main()
