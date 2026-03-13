"""
textract_cluster.py
--------------------
ClusterGrade AI — Real PDF Pipeline (Amazon Textract + Scikit-Learn)
=====================================================================
Pipeline:
  1. Convert every page of every PDF in --input_dir to a PNG image
  2. Send each page image to Amazon Textract (real OCR on scanned handwriting)
  3. Parse the returned text to detect question boundaries (Q1, Q2, 1., 2. etc.)
  4. Group all answers for the SAME question across ALL students
  5. Run TF-IDF + K-Means clustering PER question
  6. Print a per-question cluster report

AWS credentials are read from environment variables — never hardcoded.
Set them in your terminal before running:
    export AWS_DEFAULT_REGION="us-west-2"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_SESSION_TOKEN="..."   # only needed for temporary/SSO credentials

Usage:
    python textract_cluster.py --input_dir sample_answers/
    python textract_cluster.py --input_dir sample_answers/ --n_clusters 3
"""

import argparse
import os
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGES_DIR = "textract_pages"          # temp folder for per-page PNGs
REPORT_PATH = "textract_report.txt"   # full text report saved here
PLOT_DIR = "textract_plots"           # one scatter plot per question

# Patterns that mark a new TOP-LEVEL question.
# Matches standalone: "1.", "2.", "3." ... "9." at or near the start of a
# text run — intentionally avoids sub-parts like "(1a)" or "1.5".
# Also matches "Problem 3", "Question 3".
QUESTION_PATTERN = re.compile(
    r"""
    (?:^|(?<=\s))                         # start or preceded by whitespace
    (?:
        (?:problem|question|prob\.?|q\.?) # optional keyword prefix
        \s*
    )?
    (?<!\()                               # NOT preceded by '(' (excludes sub-parts)
    ([1-9])                               # single digit question number 1-9
    (?=[.)\s])                            # followed by '.', ')' or whitespace
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Step 1 — PDF → per-page PNG images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, out_dir: str) -> list[str]:
    """
    Convert all pages of a PDF to PNG files.
    Returns a sorted list of image paths for this PDF.
    """
    stem = Path(pdf_path).stem
    pages = convert_from_path(pdf_path, dpi=200)
    paths: list[str] = []
    for i, page in enumerate(pages, start=1):
        fname = os.path.join(out_dir, f"{stem}_page{i:03d}.png")
        page.save(fname, "PNG")
        paths.append(fname)
    return paths


# ---------------------------------------------------------------------------
# Step 2 — Amazon Textract OCR
# ---------------------------------------------------------------------------

def textract_ocr(image_path: str, client) -> tuple[str, float]:
    """
    Send a local image to Amazon Textract and return (full_text, avg_confidence).
    Falls back to a clear error message if credentials are missing/invalid.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.detect_document_text(
        Document={"Bytes": img_bytes}
    )

    words, confs = [], []
    for block in response.get("Blocks", []):
        if block["BlockType"] == "WORD":
            words.append(block["Text"])
            confs.append(block.get("Confidence", 100.0))

    full_text = " ".join(words)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return full_text, avg_conf


# ---------------------------------------------------------------------------
# Step 3 — Parse questions out of a page's text
# ---------------------------------------------------------------------------

def parse_questions(text: str) -> dict[str, str]:
    """
    Detect top-level question boundaries in OCR text and return a dict:
        { "Q1": "answer text...", "Q2": "answer text...", ... }

    Strategy:
    - Scan for standalone question numbers (1–9) appearing as "1." / "2." etc.
    - Keep only the FIRST occurrence of each question number (dedup).
    - If NO markers found, store the whole page text as "Q_all".
    """
    all_matches = list(QUESTION_PATTERN.finditer(text))
    if not all_matches:
        return {"Q_all": text.strip()}

    # Deduplicate: first occurrence of each question number only
    seen: set[str] = set()
    unique: list[tuple[str, int, int]] = []  # (q_num, match_start, match_end)
    for m in all_matches:
        q_num = m.group(1)
        if q_num not in seen:
            seen.add(q_num)
            unique.append((q_num, m.start(), m.end()))

    if not unique:
        return {"Q_all": text.strip()}

    questions: dict[str, str] = {}
    for idx, (q_num, _mstart, mend) in enumerate(unique):
        label = f"Q{q_num}"
        # Answer text runs from end of this marker to start of next marker
        next_start = unique[idx + 1][1] if idx + 1 < len(unique) else len(text)
        questions[label] = text[mend:next_start].strip()

    return questions


# ---------------------------------------------------------------------------
# Step 4 — Cluster answers for a single question
# ---------------------------------------------------------------------------

def pick_best_k(matrix: np.ndarray, k_min: int = 2, k_max: int = 5) -> tuple[int, float]:
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return k_min, 0.0

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def cluster_question(
    q_label: str,
    submissions: list[tuple[str, str]],   # [(student_id, answer_text), ...]
    n_clusters: int = 0,
) -> dict:
    """
    Cluster all submissions for a single question.
    Returns a result dict ready for reporting.
    """
    ids = [s[0] for s in submissions]
    texts = [s[1] if s[1].strip() else "[blank]" for s in submissions]

    if len(set(texts)) == 1:
        # All identical — one cluster
        return {
            "question": q_label,
            "n_submissions": len(submissions),
            "clusters": [{"id": 1, "members": ids, "representative": ids[0],
                           "preview": textwrap.shorten(texts[0], 80)}],
            "silhouette": 1.0,
            "note": "All submissions identical — single cluster.",
        }

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        matrix = vectorizer.fit_transform(texts).toarray()
    except ValueError:
        return {
            "question": q_label,
            "n_submissions": len(submissions),
            "clusters": [{"id": 1, "members": ids, "representative": ids[0],
                           "preview": "[vectorization failed]"}],
            "silhouette": 0.0,
            "note": "Vectorization failed — possible empty answers.",
        }

    if n_clusters > 0:
        k = min(n_clusters, len(submissions) - 1)
        sil = 0.0
    else:
        k, sil = pick_best_k(matrix)

    if k < 2:
        labels = np.zeros(len(submissions), dtype=int)
        centers = matrix.mean(axis=0, keepdims=True)
        sil = 0.0
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        centers = km.cluster_centers_
        n_unique_labels = len(set(labels.tolist()))
        # silhouette needs: 2 <= n_labels <= n_samples - 1
        if n_unique_labels >= 2 and n_unique_labels < len(submissions):
            sil = silhouette_score(matrix, labels)
        else:
            sil = 0.0

    clusters = []
    for cid in range(k):
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        if not idxs:
            continue
        # representative = submission closest to centroid
        dists = np.linalg.norm(matrix[idxs] - centers[cid], axis=1)
        rep_idx = idxs[int(np.argmin(dists))]
        clusters.append({
            "id": cid + 1,
            "members": [ids[i] for i in idxs],
            "representative": ids[rep_idx],
            "preview": textwrap.shorten(texts[rep_idx].replace("\n", " "), 80),
        })

    return {
        "question": q_label,
        "n_submissions": len(submissions),
        "clusters": clusters,
        "silhouette": sil,
        "matrix": matrix,
        "labels": labels,
        "ids": ids,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Step 5 — Reporting
# ---------------------------------------------------------------------------

def format_report(results: list[dict]) -> str:
    sep = "=" * 66
    lines = [sep, "  ClusterGrade AI — Per-Question Cluster Report", sep, ""]

    for r in results:
        lines.append(f"{'─' * 66}")
        lines.append(f"  {r['question']}  ({r['n_submissions']} submission(s))")
        if r["note"]:
            lines.append(f"  Note: {r['note']}")
        lines.append("")

        for c in r["clusters"]:
            lines.append(f"  CLUSTER {c['id']}  ({len(c['members'])} student(s))")
            lines.append(f"    Representative : {c['representative']}")
            lines.append(f"    Preview        : \"{c['preview']}\"")
            lines.append(f"    Members        : {', '.join(c['members'])}")
            lines.append("")

        sil = r.get("silhouette", 0.0)
        lines.append(f"  Silhouette Score : {sil:.4f}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def save_scatter(result: dict, out_dir: str) -> None:
    """Save a 2-D PCA scatter plot for one question's clusters."""
    matrix = result.get("matrix")
    labels = result.get("labels")
    ids = result.get("ids")
    if matrix is None or len(set(labels)) < 2:
        return

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    n_clusters = len(set(labels))
    cmap = plt.colormaps.get_cmap("tab10").resampled(n_clusters)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=cmap(int(labels[i])), s=90, zorder=3)
        ax.annotate(ids[i], (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    patches = [mpatches.Patch(color=cmap(k), label=f"Cluster {k + 1}")
               for k in range(n_clusters)]
    ax.legend(handles=patches)
    ax.set_title(f"ClusterGrade AI — {result['question']} (PCA 2-D)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, f"{result['question']}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClusterGrade AI — Textract PDF pipeline"
    )
    parser.add_argument(
        "--input_dir", default="sample_answers",
        help="Folder containing student PDF files (default: sample_answers/)"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=0,
        help="Clusters per question. 0 = auto-detect (default)."
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip saving scatter plots."
    )
    args = parser.parse_args()

    # ── Validate AWS credentials are set ──────────────────────────────────
    required_env = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"Error: Missing AWS environment variables: {', '.join(missing)}\n"
            "Set them in your terminal before running:\n"
            "  export AWS_DEFAULT_REGION='us-west-2'\n"
            "  export AWS_ACCESS_KEY_ID='...'\n"
            "  export AWS_SECRET_ACCESS_KEY='...'\n"
            "  export AWS_SESSION_TOKEN='...'   # if using temporary credentials"
        )

    textract = boto3.client("textract")

    # ── Find PDFs ─────────────────────────────────────────────────────────
    pdf_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        sys.exit(f"No PDF files found in '{args.input_dir}'.")

    print(f"Found {len(pdf_files)} PDF(s): {[os.path.basename(p) for p in pdf_files]}\n")

    # ── Convert PDFs to images ────────────────────────────────────────────
    os.makedirs(PAGES_DIR, exist_ok=True)
    print("Step 1 — Converting PDF pages to images...")
    all_page_images: list[tuple[str, str]] = []   # (student_id, image_path)
    for pdf in pdf_files:
        student_id = Path(pdf).stem
        print(f"  {os.path.basename(pdf)}: ", end="", flush=True)
        imgs = pdf_to_images(pdf, PAGES_DIR)
        print(f"{len(imgs)} page(s)")
        for img in imgs:
            all_page_images.append((student_id, img))

    # ── Textract OCR each page ────────────────────────────────────────────
    print(f"\nStep 2 — Running Amazon Textract OCR on {len(all_page_images)} page(s)...")
    # per_student_questions[student_id][q_label] accumulates answer text
    per_student: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    low_confidence_flags: list[str] = []

    for student_id, img_path in all_page_images:
        page_label = os.path.basename(img_path)
        print(f"  OCR: {page_label} ... ", end="", flush=True)
        try:
            text, avg_conf = textract_ocr(img_path, textract)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        flag = " ⚠ low confidence" if avg_conf < 60 else ""
        print(f"conf={avg_conf:.1f}%{flag}")
        if avg_conf < 60:
            low_confidence_flags.append(page_label)

        # Parse question sections from this page
        questions = parse_questions(text)
        for q_label, answer in questions.items():
            per_student[student_id][q_label].append(answer)

    # ── Aggregate: question → list of (student_id, combined_answer) ───────
    print("\nStep 3 — Aggregating answers by question...")
    question_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for student_id, q_dict in per_student.items():
        for q_label, parts in q_dict.items():
            combined = " ".join(parts)
            question_map[q_label].append((student_id, combined))

    if not question_map:
        sys.exit("No text could be extracted. Check your PDFs and AWS credentials.")

    all_questions = sorted(question_map.keys(),
                           key=lambda x: int(re.sub(r"\D", "", x) or 0))
    print(f"  Detected questions: {all_questions}")
    for q in all_questions:
        print(f"  {q}: {len(question_map[q])} submission(s)")

    # ── Cluster per question ───────────────────────────────────────────────
    print("\nStep 4 — Clustering answers per question...")
    results = []
    for q_label in all_questions:
        subs = question_map[q_label]
        print(f"  Clustering {q_label} ({len(subs)} submissions)...")
        result = cluster_question(q_label, subs, n_clusters=args.n_clusters)
        results.append(result)

    # ── Report ─────────────────────────────────────────────────────────────
    report = format_report(results)
    print("\n" + report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nFull report saved → {REPORT_PATH}")

    if low_confidence_flags:
        print(f"\n⚠  Low-confidence pages flagged for human review:")
        for p in low_confidence_flags:
            print(f"   • {p}")

    # ── Scatter plots ──────────────────────────────────────────────────────
    if not args.no_plot:
        os.makedirs(PLOT_DIR, exist_ok=True)
        print(f"\nSaving cluster scatter plots to '{PLOT_DIR}/'...")
        for r in results:
            save_scatter(r, PLOT_DIR)


if __name__ == "__main__":
    main()
