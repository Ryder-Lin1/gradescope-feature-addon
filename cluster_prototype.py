"""
cluster_prototype.py
---------------------
ClusterGrade AI — Local Clustering Prototype
============================================================
Pipeline:
  1. Read every image in --input_dir
  2. Extract text via Tesseract OCR  (pytesseract)
  3. Vectorize with TF-IDF
  4. Choose the best K automatically (silhouette score) or use --n_clusters
  5. Run K-Means clustering
  6. Print a cluster report: representative answer + all members per cluster
  7. (Optional) Save a 2-D PCA scatter plot of the clusters

Usage:
    python cluster_prototype.py --input_dir sample_answers/ --n_clusters 3

Requirements:
    pip install -r requirements.txt
    brew install tesseract        # macOS — provides the OCR engine
"""

import argparse
import os
import sys
import textwrap

import numpy as np
from PIL import Image
import pytesseract
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


def load_images(input_dir: str) -> dict[str, str]:
    """
    Walk input_dir and OCR every image file found.
    Returns a dict mapping filename -> extracted text.
    """
    if not os.path.isdir(input_dir):
        sys.exit(f"Error: '{input_dir}' is not a directory.")

    results: dict[str, str] = {}
    files = sorted(
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not files:
        sys.exit(f"Error: no image files found in '{input_dir}'.")

    print(f"Found {len(files)} image(s). Running OCR...\n")
    for fname in files:
        path = os.path.join(input_dir, fname)
        try:
            img = Image.open(path)
            text = pytesseract.image_to_string(img).strip()
        except Exception as exc:
            print(f"  [WARN] Could not process {fname}: {exc}")
            text = ""
        results[fname] = text if text else "[no text extracted]"
        # Show a short preview
        preview = text.replace("\n", " ")[:60]
        print(f"  {fname}: \"{preview}...\"")

    return results


def pick_best_k(matrix: np.ndarray, k_min: int = 2, k_max: int = 6) -> int:
    """
    Try K values from k_min to min(k_max, n_samples-1) and return the K with
    the highest silhouette score.
    """
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return k_min

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k, best_score = k, score

    print(f"\nOptimal cluster range tested: {k_min} – {k_max}")
    print(f"Best K = {best_k}  (silhouette score: {best_score:.4f})")
    return best_k


def find_representative(
    cluster_indices: list[int],
    matrix: np.ndarray,
    center: np.ndarray,
) -> int:
    """Return the index (into cluster_indices) closest to the cluster centroid."""
    dists = np.linalg.norm(matrix[cluster_indices] - center, axis=1)
    return cluster_indices[int(np.argmin(dists))]


def print_report(
    filenames: list[str],
    texts: list[str],
    labels: np.ndarray,
    centers: np.ndarray,
    matrix: np.ndarray,
    silhouette: float,
) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  ClusterGrade AI — Clustering Report")
    print(sep)
    print(f"  Processed {len(filenames)} submissions.\n")

    n_clusters = centers.shape[0]
    for cluster_id in range(n_clusters):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        rep_idx = find_representative(indices, matrix, centers[cluster_id])
        rep_name = filenames[rep_idx]
        preview = texts[rep_idx].replace("\n", " ")[:80]

        print(f"CLUSTER {cluster_id + 1}  ({len(indices)} submission{'s' if len(indices) != 1 else ''})")
        print(f"  Representative answer : {rep_name}")
        print(f"  Preview               : \"{textwrap.shorten(preview, width=70)}\"")
        print(f"  Members               : {', '.join(filenames[i] for i in indices)}")
        print()

    print(f"Silhouette Score: {silhouette:.4f}  (range: -1 → 1, higher = more distinct clusters)")
    print(sep)


def save_scatter(
    matrix: np.ndarray,
    labels: np.ndarray,
    filenames: list[str],
    output_path: str = "cluster_plot.png",
) -> None:
    """Project to 2-D with PCA and save a labelled scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    n_clusters = len(set(labels))
    colors = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=colors(labels[i]), s=80, zorder=3)
        ax.annotate(
            filenames[i],
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    patches = [
        mpatches.Patch(color=colors(k), label=f"Cluster {k + 1}")
        for k in range(n_clusters)
    ]
    ax.legend(handles=patches, loc="best")
    ax.set_title("ClusterGrade AI — Submission Clusters (PCA 2-D projection)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nScatter plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClusterGrade AI — OCR + clustering prototype"
    )
    parser.add_argument(
        "--input_dir",
        default="sample_answers",
        help="Folder of student answer images (default: sample_answers/)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=0,
        help="Number of clusters. Set to 0 (default) to auto-detect via silhouette score.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip saving the 2-D PCA scatter plot.",
    )
    args = parser.parse_args()

    # Step 1 — OCR
    file_text_map = load_images(args.input_dir)
    filenames = list(file_text_map.keys())
    texts = list(file_text_map.values())

    if len(filenames) < 2:
        sys.exit("Need at least 2 submissions to cluster.")

    # Step 2 — Vectorise with TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    # Step 3 — Choose K
    if args.n_clusters > 0:
        k = min(args.n_clusters, len(filenames) - 1)
        print(f"\nUsing user-specified n_clusters = {k}")
    else:
        k = pick_best_k(tfidf_matrix)

    # Step 4 — K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)
    centers = km.cluster_centers_

    # Step 5 — Evaluate
    if len(set(labels)) > 1:
        sil = silhouette_score(tfidf_matrix, labels)
    else:
        sil = 0.0

    # Step 6 — Report
    print_report(filenames, texts, labels, centers, tfidf_matrix, sil)

    # Step 7 — (Optional) scatter plot
    if not args.no_plot:
        save_scatter(tfidf_matrix, labels, filenames)


if __name__ == "__main__":
    main()
