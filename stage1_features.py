import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import csv
import os

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION FUNCTIONS
# ─────────────────────────────────────────────

def get_density_feature(gray_filtered, threshold):
    """Cloud density: % of pixels brighter than Otsu threshold."""
    bright_pixels = np.sum(gray_filtered > threshold)
    return (bright_pixels / gray_filtered.size) * 100


def get_texture_features(gray):
    """
    GLCM texture features — describe how 'rough' or 'smooth' the cloud looks.
    - Contrast    : large value = big differences between neighbours (fluffy/jagged clouds)
    - Homogeneity : large value = very uniform texture (smooth stratus)
    - Energy      : large value = very regular/repetitive pattern
    - Correlation : how linearly related neighbouring pixels are
    """
    # Reduce to 64 grey levels so GLCM is faster and more stable on small images
    gray_reduced = (gray // 4).astype(np.uint8)          # 0-255 → 0-63
    glcm = graycomatrix(gray_reduced, distances=[1],
                        angles=[0], levels=64, symmetric=True, normed=True)
    contrast     = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity  = graycoprops(glcm, 'homogeneity')[0, 0]
    energy       = graycoprops(glcm, 'energy')[0, 0]
    correlation  = graycoprops(glcm, 'correlation')[0, 0]
    return contrast, homogeneity, energy, correlation


def get_edge_density(gray):
    """
    Edge density: % of pixels that are edges (Canny).
    High value = lots of edges = fluffy cumulus.
    Low value  = smooth, few edges = stratus or clear sky.
    """
    edges = cv2.Canny(gray, 50, 150)
    return (np.sum(edges > 0) / edges.size) * 100


def get_blue_ratio(image_bgr):
    """
    Blue channel ratio.
    Clear sky = high blue ratio.
    Thick cloud = all channels similar (white/grey) → ratio closer to 0.33.
    """
    b = image_bgr[:, :, 0].astype(float)
    g = image_bgr[:, :, 1].astype(float)
    r = image_bgr[:, :, 2].astype(float)
    total = b + g + r + 1e-6          # avoid division by zero
    return np.mean(b / total)


def get_brightness_std(gray):
    """Standard deviation of brightness — high = varied lighting = complex cloud scene."""
    return np.std(gray)


def auto_label(density):
    """
    Generate a label from density (same logic as your original classify_cloud_type).
    This is our 'cheap' labeller — good enough to start training.
    You can manually override labels in the CSV later.
    """
    if density < 5:
        return "Clear"
    elif density < 20:
        return "Cirrus"
    elif density < 40:
        return "Cumulus"
    elif density < 60:
        return "Stratus"
    else:
        return "Overcast"


# ─────────────────────────────────────────────
#  PROCESS ALL IMAGES
# ─────────────────────────────────────────────

def extract_features_from_image(image_path):
    """Run the full pipeline on one image and return a feature dict."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ⚠  Could not load: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Same brightness filter from your original code
    bright_threshold = np.percentile(gray, 98)
    gray_filtered = gray.copy()
    gray_filtered[gray > bright_threshold] = np.median(gray)

    # Otsu threshold on filtered image
    otsu_thresh, _ = cv2.threshold(gray_filtered, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Extract all features ──
    density     = get_density_feature(gray_filtered, otsu_thresh)
    contrast, homogeneity, energy, correlation = get_texture_features(gray_filtered)
    edge_density   = get_edge_density(gray_filtered)
    blue_ratio     = get_blue_ratio(image)
    brightness_std = get_brightness_std(gray_filtered)
    label          = auto_label(density)

    return {
        "image":          os.path.basename(image_path),
        "density":        round(density, 4),
        "contrast":       round(contrast, 4),
        "homogeneity":    round(homogeneity, 4),
        "energy":         round(energy, 4),
        "correlation":    round(correlation, 4),
        "edge_density":   round(edge_density, 4),
        "blue_ratio":     round(blue_ratio, 4),
        "brightness_std": round(brightness_std, 4),
        "label":          label,
    }


def process_all_images(image_folder="images", output_csv="features.csv"):
    results = []

    for i in range(0, 26):                           # pic_1.jpg … pic_16.jpg
        path = os.path.join(image_folder, f"pic_{i}.jpg")
        if not os.path.exists(path):
            print(f"  Skipping (not found): {path}")
            continue

        print(f"Processing {path} ...", end="  ")
        feat = extract_features_from_image(path)
        if feat:
            results.append(feat)
            print(f"density={feat['density']:.1f}%  label={feat['label']}")

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅  Saved features for {len(results)} images → {output_csv}")
    else:
        print("❌  No images were processed.")

    return results


# ─────────────────────────────────────────────
#  QUICK VISUAL CHECK
# ─────────────────────────────────────────────

def plot_feature_overview(results):
    """Bar chart of each feature per image — lets you spot outliers."""
    if not results:
        return

    images   = [r["image"] for r in results]
    features = ["density", "contrast", "homogeneity", "edge_density", "blue_ratio"]
    colors   = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]

    fig, axes = plt.subplots(len(features), 1, figsize=(14, 3 * len(features)))

    for ax, feat, color in zip(axes, features, colors):
        values = [r[feat] for r in results]
        bars = ax.bar(images, values, color=color, alpha=0.8)
        ax.set_title(feat.replace("_", " ").title(), fontsize=11, weight="bold")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        # Annotate bars with auto-label
        for bar, res in zip(bars, results):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    res["label"][0],          # first letter of label
                    ha="center", va="bottom", fontsize=7, color="black")

    plt.suptitle("Feature Overview — Letter = Auto Label (C=Clear, Ci=Cirrus, Cu=Cumulus, S=Stratus, O=Overcast)",
                 fontsize=9, y=1.01)
    plt.tight_layout()
    plt.savefig("feature_overview.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("📊  Chart saved → feature_overview.png")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Stage 1 — Feature Extraction")
    print("=" * 50)

    results = process_all_images(image_folder="images", output_csv="features.csv")
    plot_feature_overview(results)

    print("\nFeature table preview:")
    print(f"{'Image':<12} {'Density':>8} {'Contrast':>10} {'Homogen.':>10} "
          f"{'EdgeDen.':>10} {'BlueRatio':>10} {'Label'}")
    print("-" * 70)
    for r in results:
        print(f"{r['image']:<12} {r['density']:>8.2f} {r['contrast']:>10.4f} "
              f"{r['homogeneity']:>10.4f} {r['edge_density']:>10.4f} "
              f"{r['blue_ratio']:>10.4f}  {r['label']}")
