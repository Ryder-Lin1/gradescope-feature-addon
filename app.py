"""
app.py — ClusterGrade AI  •  Gradescope Feature Add-on
=======================================================
Runs as a sidecar server alongside Gradescope.
Integration methods:
  1. Browser Bookmarklet  — injects a floating panel into any Gradescope page
  2. Direct URL           — http://localhost:5000  (standalone dashboard)
  3. iFrame embed         — /embed URL for LMS / course pages

Set AWS env vars before running:
  export AWS_DEFAULT_REGION="us-west-2"
  export AWS_ACCESS_KEY_ID="..."
  export AWS_SECRET_ACCESS_KEY="..."
  export AWS_SESSION_TOKEN="..."
  python app.py
"""

import base64
import os
import re
import sys
import threading
import traceback
import uuid
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
from pdf2image import convert_from_path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

UPLOAD_DIR = "uploads"
PAGES_DIR = "textract_pages"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PAGES_DIR, exist_ok=True)

CLUSTER_COLORS = ["#4A90D9", "#E05C5C", "#50C878", "#9B59B6",
                  "#F39C12", "#1ABC9C", "#E74C3C", "#3498DB"]

QUESTION_RE = re.compile(
    r"""(?:^|(?<=\s))(?:problem|question|prob\.?|q\.?)?\s*(?<!\()([1-9])(?=[.)\s])""",
    re.IGNORECASE | re.VERBOSE,
)

# job_id -> result dict
_jobs: dict = {}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path, job_dir):
    stem = Path(pdf_path).stem
    pages = convert_from_path(pdf_path, dpi=150)
    paths = []
    for i, page in enumerate(pages, 1):
        fname = os.path.join(job_dir, "%s_p%03d.png" % (stem, i))
        page.save(fname, "PNG")
        paths.append(fname)
    return paths


def textract_ocr(image_path, client):
    with open(image_path, "rb") as f:
        data = f.read()
    resp = client.detect_document_text(Document={"Bytes": data})
    words, confs = [], []
    for b in resp.get("Blocks", []):
        if b["BlockType"] == "WORD":
            words.append(b["Text"])
            confs.append(b.get("Confidence", 100.0))
    return " ".join(words), (float(np.mean(confs)) if confs else 0.0)


def textract_ocr_with_lines(image_path, client):
    """OCR that returns LINE blocks with geometry for box detection."""
    with open(image_path, "rb") as f:
        data = f.read()
    resp = client.detect_document_text(Document={"Bytes": data})
    lines = []
    for b in resp.get("Blocks", []):
        if b["BlockType"] == "LINE":
            geo = b.get("Geometry", {}).get("BoundingBox", {})
            lines.append({
                "text": b.get("Text", ""),
                "top": geo.get("Top", 0),
                "left": geo.get("Left", 0),
                "width": geo.get("Width", 0),
                "height": geo.get("Height", 0),
                "confidence": b.get("Confidence", 100.0),
            })
    lines.sort(key=lambda l: l["top"])
    return lines


def extract_boxed_answer(lines):
    """Detect the boxed answer region on a page using Textract geometry.

    Strategy: The answer box is typically an indented/bordered region.
    Lines inside the box tend to be narrower (don't span full page width)
    and are indented from the left margin compared to the question text.

    We detect the box by looking for lines that are indented (left > 0.08)
    or lines that appear after a significant vertical gap from the question.
    The question text is the un-indented text at the top of the page.
    """
    if not lines:
        return "", ""

    if len(lines) == 1:
        return "", lines[0]["text"]

    # Compute the most common left margin (the "normal" margin for question text)
    lefts = [l["left"] for l in lines]
    # Question lines are typically at the leftmost margin
    min_left = min(lefts)

    # Find lines that are significantly indented (inside a box)
    # or find the biggest vertical gap to split question from answer
    # Approach: combine gap detection + indentation detection

    # 1) Find biggest vertical gap
    best_gap_idx = 0
    best_gap = 0
    for i in range(len(lines) - 1):
        gap = lines[i + 1]["top"] - (lines[i]["top"] + lines[i]["height"])
        if gap > best_gap:
            best_gap = gap
            best_gap_idx = i

    # 2) Find indented lines (left margin > min_left + threshold)
    indent_threshold = 0.03  # 3% of page width indentation
    indented_start = None
    for i, l in enumerate(lines):
        if l["left"] > min_left + indent_threshold:
            indented_start = i
            break

    # Decide split point: prefer indentation if found, else use gap
    if indented_start is not None and indented_start > 0:
        split_idx = indented_start
    elif best_gap > 0.015:  # at least 1.5% page height gap
        split_idx = best_gap_idx + 1
    else:
        # No clear separation: first 1-2 lines are question, rest is answer
        split_idx = min(2, len(lines) - 1)

    q_lines = lines[:split_idx]
    a_lines = lines[split_idx:]

    question_text = " ".join(l["text"] for l in q_lines).strip()
    answer_text = " ".join(l["text"] for l in a_lines).strip()
    return question_text, answer_text


def normalize_answer(text):
    """Normalize an answer string for exact-match grouping.
    Lowercases, strips whitespace, removes extra spaces and punctuation differences.
    """
    t = text.lower().strip()
    t = re.sub(r'\s+', ' ', t)          # collapse whitespace
    t = re.sub(r'[^\w\s]', '', t)       # remove punctuation
    return t


def group_by_answer(subs):
    """Group submissions by exact boxed answer text.

    subs: list of (student_id, answer_text)
    Returns a dict similar to cluster_answers output but grouped by identical answers.
    """
    ids = [s[0] for s in subs]
    texts = [s[1] for s in subs]
    n = len(subs)

    # Group by normalized answer
    groups = defaultdict(list)
    for i, (sid, ans) in enumerate(subs):
        key = normalize_answer(ans) if ans.strip() else "[blank]"
        groups[key].append(i)

    clusters = []
    for gid, (norm_ans, idxs) in enumerate(sorted(groups.items(),
                                                    key=lambda x: -len(x[1]))):
        # The representative is the first student in the group
        ri = idxs[0]
        raw_answer = texts[ri] if texts[ri].strip() else "[blank]"
        clusters.append({
            "id": gid + 1,
            "color": CLUSTER_COLORS[gid % len(CLUSTER_COLORS)],
            "representative": ids[ri],
            "answer": raw_answer,
            "members": [
                {"id": ids[i], "preview": texts[i][:200] if texts[i].strip() else "[blank]",
                 "is_rep": i == ri}
                for i in idxs
            ],
        })

    # Simple similarity: 1.0 for same group, 0.0 for different
    label_map = {}
    for c in clusters:
        for m in c["members"]:
            label_map[m["id"]] = c["id"]
    labels = [label_map.get(sid, 0) for sid in ids]

    sim = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sim[i][j] = 1.0 if labels[i] == labels[j] else 0.0

    # PCA coordinates — spread groups apart
    pca_pts = []
    group_x = {}
    for c in clusters:
        gx = (c["id"] - 1) * 2.0
        group_x[c["id"]] = gx
    for i, sid in enumerate(ids):
        gid = label_map.get(sid, 1)
        # Spread members within a group slightly
        offset = (i % 5) * 0.3
        pca_pts.append({"id": sid, "x": group_x.get(gid, 0) + offset,
                        "y": float(i % 3) * 0.5, "cluster": gid})

    return {
        "clusters": clusters,
        "silhouette": 1.0 if len(clusters) > 1 else 0.0,
        "sim_matrix": sim,
        "ids": ids,
        "pca_points": pca_pts,
    }


def parse_questions(text):
    matches = list(QUESTION_RE.finditer(text))
    if not matches:
        return {"Q (all)": text.strip()}
    seen, unique = set(), []
    for m in matches:
        n = m.group(1)
        if n not in seen:
            seen.add(n)
            unique.append((n, m.start(), m.end()))
    out = {}
    for idx, (q_num, _ms, mend) in enumerate(unique):
        ns = unique[idx + 1][1] if idx + 1 < len(unique) else len(text)
        out["Q%s" % q_num] = text[mend:ns].strip()
    return out


def sim_matrix(texts):
    vec = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        mat = vec.fit_transform(texts).toarray()
        sim = cosine_similarity(mat)
        return [[round(float(v), 3) for v in row] for row in sim]
    except Exception:
        n = len(texts)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def cluster_answers(subs, n_clusters=0):
    ids = [s[0] for s in subs]
    texts = [s[1] if s[1].strip() else "[blank]" for s in subs]
    n = len(subs)

    vec = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        matrix = vec.fit_transform(texts).toarray()
    except Exception:
        return {
            "clusters": [{"id": 1, "color": CLUSTER_COLORS[0],
                          "members": [{"id": ids[i], "preview": texts[i][:150],
                                       "is_rep": i == 0} for i in range(n)],
                          "representative": ids[0]}],
            "silhouette": 0.0, "sim_matrix": [], "ids": ids, "pca_points": [],
        }

    if n_clusters > 0:
        k = min(n_clusters, max(1, n - 1))
    else:
        best_k, best_sil = 1, -1.0
        for kt in range(2, min(6, n)):
            km = KMeans(n_clusters=kt, random_state=42, n_init=10)
            lb = km.fit_predict(matrix)
            nu = len(set(lb.tolist()))
            s = silhouette_score(matrix, lb) if 1 < nu < n else 0.0
            if s > best_sil:
                best_k, best_sil = kt, s
        k = best_k

    if k < 2 or n < 3:
        labels = np.arange(n)
        centers = matrix
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        centers = km.cluster_centers_

    nu = len(set(labels.tolist()))
    sil = silhouette_score(matrix, labels) if 1 < nu < n else 0.0

    # PCA for scatter plot — handle edge cases safely
    coords = []
    if n >= 2 and matrix.shape[1] >= 2:
        try:
            coords = PCA(n_components=2, random_state=42).fit_transform(matrix).tolist()
        except Exception:
            coords = [[0.0, 0.0]] * n
    elif n >= 2 and matrix.shape[1] == 1:
        coords = [[float(matrix[i, 0]), 0.0] for i in range(n)]
    else:
        coords = [[0.0, 0.0]] * n
    # Ensure every coord has exactly 2 values
    coords = [(c + [0.0, 0.0])[:2] for c in coords]

    clusters = []
    for cid in range(int(labels.max()) + 1):
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        if not idxs:
            continue
        if len(centers.shape) == 1:
            center_vec = centers
        else:
            center_vec = centers[min(cid, len(centers) - 1)]
        try:
            dists = np.linalg.norm(matrix[idxs] - center_vec, axis=1)
            ri = idxs[int(np.argmin(dists))]
        except Exception:
            ri = idxs[0]
        clusters.append({
            "id": cid + 1,
            "color": CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            "representative": ids[ri],
            "members": [
                {"id": ids[i], "preview": texts[i][:200], "is_rep": i == ri}
                for i in idxs
            ],
        })

    return {
        "clusters": clusters,
        "silhouette": round(sil, 4),
        "sim_matrix": sim_matrix(texts),
        "ids": ids,
        "pca_points": [
            {"id": ids[i], "x": float(coords[i][0]), "y": float(coords[i][1]),
             "cluster": int(labels[i]) + 1}
            for i in range(n)
        ],
    }


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_pipeline(pdf_paths, job_id):
    """Each page of the PDF is treated as one question.
    AWS Textract OCRs each page, extracts the boxed answer,
    then groups students with the same boxed answer together.
    """
    _jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting..."}
    try:
        textract = boto3.client("textract")
        job_dir = os.path.join(PAGES_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        _jobs[job_id]["message"] = "Converting PDFs to images..."
        student_pages = {}          # sid -> [img_path, ...]
        total_pages = 0
        for pdf in pdf_paths:
            sid = Path(pdf).stem
            imgs = pdf_to_images(pdf, job_dir)
            student_pages[sid] = imgs
            total_pages += len(imgs)

        # Determine number of pages (use max across students)
        max_pages = max(len(imgs) for imgs in student_pages.values())

        # page_data[page_idx] = list of (sid, answer_text, question_text, img_path)
        page_data = defaultdict(list)
        done = 0

        for sid, imgs in student_pages.items():
            n_pages = len(imgs)
            for pg_i, img in enumerate(imgs):
                _jobs[job_id]["message"] = (
                    "AWS Textract OCR: %s (page %d/%d)" % (sid, pg_i + 1, n_pages)
                )
                _jobs[job_id]["progress"] = int(50 * done / max(total_pages, 1))

                lines = textract_ocr_with_lines(img, textract)
                _q_text, a_text = extract_boxed_answer(lines)

                page_data[pg_i].append((sid, a_text, _q_text, img))
                done += 1

        # Build question results — one per page
        questions_data = []
        for pg_i in range(max_pages):
            subs = page_data.get(pg_i, [])
            if not subs:
                continue

            q_label = "Page %d" % (pg_i + 1)

            _jobs[job_id]["message"] = "Grouping answers for %s..." % q_label
            _jobs[job_id]["progress"] = 50 + int(50 * pg_i / max(max_pages, 1))

            # Group identical boxed answers together
            answer_subs = [(sid, a_text) for sid, a_text, _, _ in subs]
            result = group_by_answer(answer_subs)
            result["question"] = q_label
            result["question_text"] = ""  # only show boxed final answers
            result["n_submissions"] = len(subs)

            # Attach page images to each member
            img_lookup = {sid: img for sid, _, _, img in subs}
            for cluster in result["clusters"]:
                for member in cluster["members"]:
                    ipath = img_lookup.get(member["id"], "")
                    member["image_b64"] = img_b64(ipath) if ipath else ""

            questions_data.append(result)

        _jobs[job_id] = {
            "status": "done", "progress": 100, "message": "Complete",
            "questions": questions_data,
            "students": list(student_pages.keys()),
            "pdfs": [os.path.basename(p) for p in pdf_paths],
        }
    except Exception as e:
        _jobs[job_id] = {
            "status": "error", "message": str(e),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/embed")
def embed():
    style = "<style>header{display:none!important}body{background:#f5f7fa!important}</style>"
    return Response(HTML.replace("</head>", style + "</head>"), mimetype="text/html")


@app.route("/bookmarklet.js")
def bookmarklet_js():
    js = ("(function(){"
          "if(document.getElementById('cgai-panel')){"
          "document.getElementById('cgai-panel').remove();return;}"
          "var p=document.createElement('div');"
          "p.id='cgai-panel';"
          "p.style='position:fixed;right:0;top:0;width:480px;height:100vh;"
          "z-index:99999;box-shadow:-4px 0 24px rgba(0,0,0,.3)';"
          "var f=document.createElement('iframe');"
          "f.src='http://localhost:5000/embed';"
          "f.style='width:100%;height:100%;border:none;';"
          "p.appendChild(f);document.body.appendChild(p);"
          "})();")
    return js, 200, {"Content-Type": "application/javascript"}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("pdfs")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded"}), 400
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    paths = []
    for f in files:
        fname = secure_filename(f.filename)
        path = os.path.join(job_dir, fname)
        f.save(path)
        paths.append(path)
    threading.Thread(target=run_pipeline, args=(paths, job_id), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = _jobs.get(job_id, {"status": "unknown"})
    if job.get("status") == "done":
        return jsonify(job)
    return jsonify({k: v for k, v in job.items() if k != "questions"})


@app.route("/api/sample")
def api_sample():
    pdfs = sorted(
        os.path.join("sample_answers", f)
        for f in os.listdir("sample_answers")
        if f.lower().endswith(".pdf")
    )
    if not pdfs:
        return jsonify({"error": "No PDFs in sample_answers/"}), 400
    job_id = "sample"
    if _jobs.get(job_id, {}).get("status") == "done":
        return jsonify({"job_id": job_id})
    _jobs.pop(job_id, None)
    threading.Thread(target=run_pipeline, args=(pdfs, job_id), daemon=True).start()
    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# HTML — Gradescope-style UI
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ClusterGrade AI</title>
<style>
:root{--blue:#0075db;--blue-dk:#005baa;--bg:#f5f7fa;--white:#fff;
  --border:#dde1e9;--text:#1a1a2e;--text2:#5a6080;--green:#27ae60;
  --yellow:#f39c12;--red:#e74c3c;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:14px}
/* ── header ── */
header{background:var(--white);border-bottom:1px solid var(--border);padding:0 28px;
  height:60px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100;
  box-shadow:0 1px 4px rgba(0,0,0,.06)}
.logo{width:36px;height:36px;background:var(--blue);border-radius:8px;display:flex;
  align-items:center;justify-content:center;color:#fff;font-size:18px}
.logo-name{font-size:17px;font-weight:700}
.hdr-r{margin-left:auto;display:flex;align-items:center;gap:10px}
.hdr-info{font-size:12px;color:var(--text2)}
/* ── buttons ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;
  padding:9px 20px;border-radius:7px;font-size:14px;font-weight:600;cursor:pointer;
  border:1.5px solid transparent;background:none;transition:all .15s}
.btn-blue{background:var(--blue);color:#fff;border-color:var(--blue)}
.btn-blue:hover{background:var(--blue-dk);border-color:var(--blue-dk)}
.btn-green{background:var(--green);color:#fff;border-color:var(--green)}
.btn-green:hover{background:#219150;border-color:#219150}
.btn-ghost{background:var(--white);color:var(--text);border-color:var(--border)}
.btn-ghost:hover{background:var(--bg)}
.btn-sm{padding:6px 14px;font-size:12px;font-weight:500}
/* ── pages ── */
.page{display:none!important;min-height:calc(100vh - 60px)}
/* ── upload page ── */
#page-upload.active{display:flex!important;flex-direction:column;align-items:center;padding:48px 24px}
.hero{text-align:center;margin-bottom:40px}
.hero h1{font-size:28px;font-weight:800;margin-bottom:8px}
.hero p{color:var(--text2);font-size:15px}
.upload-card{background:var(--white);border-radius:14px;
  box-shadow:0 2px 16px rgba(0,0,0,.08);padding:36px;width:100%;max-width:560px}
.drop-zone{border:2px dashed var(--border);border-radius:10px;padding:32px 20px;
  text-align:center;cursor:pointer;transition:all .2s;margin-bottom:20px}
.drop-zone:hover,.drop-zone.drag{border-color:var(--blue);background:#f0f6ff}
.drop-icon{font-size:36px;margin-bottom:10px}
.drop-title{font-size:15px;font-weight:600;margin-bottom:6px}
.drop-sub{color:var(--text2);font-size:13px;margin-bottom:16px}
#flist{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;min-height:0}
.fc{background:#eef4ff;border:1px solid #c4d9f8;color:var(--blue);
  font-size:12px;padding:4px 10px;border-radius:20px;display:flex;align-items:center;gap:4px}
.fc-x{cursor:pointer;color:#999;font-size:14px;line-height:1}
.fc-x:hover{color:var(--red)}
#fcount{font-size:13px;color:var(--text2);margin-bottom:16px;min-height:20px;text-align:center}
.upload-actions{display:flex;gap:10px}
.upload-actions .btn{flex:1}
.divider{display:flex;align-items:center;gap:10px;margin:20px 0;color:var(--text2);font-size:12px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:var(--border)}
.aws-badge{display:flex;align-items:center;gap:8px;background:#f0f6ff;
  border:1px solid #c4d9f8;border-radius:8px;padding:10px 14px;font-size:12px;color:var(--text2);margin-top:16px}
.aws-badge span{font-size:18px}
/* ── progress page ── */
#page-progress.active{display:flex!important;align-items:center;justify-content:center}
.prog-card{background:var(--white);border-radius:14px;box-shadow:0 2px 16px rgba(0,0,0,.08);
  padding:40px 48px;text-align:center;width:100%;max-width:440px}
.spin{width:48px;height:48px;border:4px solid var(--border);border-top-color:var(--blue);
  border-radius:50%;animation:sp .8s linear infinite;margin:0 auto 20px}
@keyframes sp{to{transform:rotate(360deg)}}
.prog-title{font-size:18px;font-weight:700;margin-bottom:6px}
.prog-msg{color:var(--text2);font-size:13px;margin-bottom:16px;min-height:20px}
.prog-bar{background:var(--bg);border-radius:100px;height:10px;overflow:hidden}
.prog-fill{background:var(--blue);height:100%;border-radius:100px;transition:width .4s;width:0%}
.prog-steps{margin-top:20px;text-align:left}
.step{display:flex;align-items:center;gap:10px;padding:6px 0;font-size:13px;color:var(--text2)}
.step.done{color:var(--green)} .step.active{color:var(--blue);font-weight:600}
.step-dot{width:8px;height:8px;border-radius:50%;background:var(--border);flex-shrink:0}
.step.done .step-dot{background:var(--green)} .step.active .step-dot{background:var(--blue)}
/* ── results page ── */
#page-results.active{display:flex!important;height:calc(100vh - 60px)}
aside{width:220px;min-width:220px;background:var(--white);
  border-right:1px solid var(--border);overflow-y:auto;flex-shrink:0}
.aside-lbl{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;
  letter-spacing:.8px;padding:16px 16px 8px}
.qbtn{display:flex;align-items:center;width:100%;background:none;border:none;
  color:var(--text2);font-size:13px;padding:9px 14px;cursor:pointer;gap:8px;
  border-left:3px solid transparent;text-align:left;transition:all .12s}
.qbtn:hover{background:#f0f4ff;color:var(--text)}
.qbtn.on{background:#eef4ff;color:var(--blue);border-left-color:var(--blue);font-weight:600}
.qnum{width:24px;height:24px;border-radius:50%;background:var(--border);
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}
.qbtn.on .qnum{background:var(--blue);color:#fff}
.qpill{margin-left:auto;font-size:10px;background:var(--bg);padding:2px 7px;border-radius:10px}
.qbtn.on .qpill{background:#c4d9f8;color:var(--blue)}
.res-main{flex:1;overflow-y:auto;padding:24px 28px}
/* clusters/heatmap/scatter */
.qhdr{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:20px}
.qtitle{font-size:22px;font-weight:800}
.pill{font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;
  border:1px solid var(--border);background:var(--white)}
.pill-b{border-color:#c4d9f8;color:var(--blue);background:#eef4ff}
.pill-g{border-color:#a9dbb8;color:var(--green);background:#eafaf1}
.pill-y{border-color:#fad7a0;color:var(--yellow);background:#fef9e7}
.pill-n{color:var(--text2)}
.ccard{background:var(--white);border:1px solid var(--border);
  border-radius:10px;margin-bottom:16px;overflow:hidden}
.chead{display:flex;align-items:center;gap:10px;padding:12px 16px;
  border-bottom:1px solid var(--border);background:#fafbfd}
.dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.csub{font-size:12px;color:var(--text2)}
.cact{margin-left:auto}
.apply-btn{background:var(--blue);color:#fff;border:none;padding:6px 14px;
  border-radius:5px;font-size:12px;font-weight:600;cursor:pointer}
.apply-btn:hover{background:var(--blue-dk)}
.apply-btn.done{background:var(--green);cursor:default}
/* answer box inside group card */
.ans-box{background:#f8faf8;border-bottom:1px solid var(--border);padding:14px 18px}
.ans-label{font-size:10px;font-weight:700;color:var(--green);text-transform:uppercase;
  letter-spacing:.8px;margin-bottom:5px}
.ans-body{font-size:15px;font-weight:800;color:var(--red);line-height:1.5;
  padding:10px 14px;background:var(--white);border:1.5px solid #a9dbb8;border-radius:8px;
  font-family:'Courier New',monospace;text-transform:uppercase}
/* student list inside group card */
.stu-list{padding:12px 16px}
.stu-label{font-size:10px;font-weight:700;color:var(--text2);text-transform:uppercase;
  letter-spacing:.8px;margin-bottom:8px}
.stu-card{display:inline-flex;align-items:center;gap:8px;
  background:var(--bg);border:1px solid var(--border);border-radius:8px;
  padding:6px 12px;margin:0 6px 6px 0;font-size:13px;font-weight:500}
.stu-thumb{width:40px;height:28px;border-radius:4px;overflow:hidden;flex-shrink:0}
.stu-thumb img{width:100%;height:100%;object-fit:cover;cursor:pointer}
.stu-name{color:var(--text)}
.mimg{width:100%;height:100%;object-fit:cover;cursor:pointer;background:#f0f0f0;display:block}
/* lightbox */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);
  z-index:9999;align-items:center;justify-content:center}
#lb.open{display:flex}
#lb img{max-width:90vw;max-height:90vh;border-radius:8px}
#lbx{position:absolute;top:16px;right:22px;font-size:28px;color:#fff;
  cursor:pointer;background:none;border:none}
/* error */
.err-box{background:#fdf2f2;border:1px solid #f5c6c6;border-radius:8px;
  padding:14px 18px;color:var(--red);font-size:13px;max-width:560px;width:100%;margin:10px 0}
</style>
</head>
<body>

<header>
  <div class="logo" style="cursor:pointer" onclick="showPage('upload')">📄</div>
  <span class="logo-name" style="cursor:pointer" onclick="showPage('upload')">ClusterGrade AI</span>
  <div class="hdr-r">
    <span class="hdr-info" id="hdr-info"></span>
    <button class="btn btn-ghost btn-sm" id="btn-back" style="display:none" onclick="showPage('upload')">
      &#8592; New Analysis
    </button>
  </div>
</header>

<!-- ════════════════════════════════════════════════════ PAGE: UPLOAD -->
<div id="page-upload" class="page active">
  <div class="hero">
    <h1>📄 ClusterGrade AI</h1>
    <p>Upload student exam PDFs &mdash; we OCR them with AWS Textract and cluster similar answers.</p>
  </div>
  <div class="upload-card">
    <div class="drop-zone" id="dz"
         onclick="document.getElementById('finput').click()"
         ondragover="dov(event)" ondragleave="dlv(event)" ondrop="ddp(event)">
      <div class="drop-icon">&#128196;</div>
      <div class="drop-title">Drop PDF files here</div>
      <div class="drop-sub">or click to browse — one PDF per student</div>
      <button class="btn btn-blue" style="margin-top:4px"
              onclick="event.stopPropagation();document.getElementById('finput').click()">
        &#128194; Choose Files
      </button>
    </div>
    <input type="file" id="finput" multiple accept=".pdf" onchange="onF(this.files)" style="display:none"/>
    <div id="flist"></div>
    <div id="fcount"></div>
    <div id="upload-err"></div>
    <div class="upload-actions">
      <button class="btn btn-green" id="btn-analyze" onclick="doUpload()">
        &#9658; Analyze PDFs
      </button>
      <button class="btn btn-ghost" onclick="doSample()">
        &#128300; Try Demo
      </button>
    </div>
    <div class="aws-badge">
      <span>&#9729;</span>
      <div>Powered by <strong>AWS Textract</strong> — extracts handwritten &amp; typed text from each page</div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════ PAGE: PROGRESS -->
<div id="page-progress" class="page">
  <div class="prog-card">
    <div class="spin"></div>
    <div class="prog-title">Analyzing Submissions</div>
    <div class="prog-msg" id="pmsg">Starting…</div>
    <div class="prog-bar"><div class="prog-fill" id="pfill"></div></div>
    <div class="prog-steps">
      <div class="step" id="step-upload"><div class="step-dot"></div>Uploading PDFs</div>
      <div class="step" id="step-ocr"><div class="step-dot"></div>AWS Textract OCR</div>
      <div class="step" id="step-parse"><div class="step-dot"></div>Parsing questions</div>
      <div class="step" id="step-cluster"><div class="step-dot"></div>Clustering answers</div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════ PAGE: RESULTS -->
<div id="page-results" class="page">
  <aside>
    <div class="aside-lbl">Questions</div>
    <div id="qlist"></div>
  </aside>
  <div class="res-main" id="res-main"></div>
</div>

<div id="lb"><button id="lbx" onclick="clb()">&#10005;</button><img id="lbimg" src="" alt=""/></div>

<script>
var DATA=null,CQ=0,SF=[],PT=null;

/* ─── page navigation ─── */
function showPage(name){
  document.querySelectorAll('.page').forEach(function(p){p.classList.remove('active');});
  document.getElementById('page-'+name).classList.add('active');
  document.getElementById('btn-back').style.display=(name==='results')?'inline-flex':'none';
  if(name==='upload') document.getElementById('hdr-info').textContent='';
}

/* ─── file selection ─── */
function onF(files){
  Array.from(files).forEach(function(f){
    if(!SF.some(function(x){return x.name===f.name&&x.size===f.size;})) SF.push(f);
  });
  renderFileList();
  document.getElementById('finput').value='';
}
function renderFileList(){
  document.getElementById('flist').innerHTML=SF.map(function(f,i){
    return'<span class="fc">'+esc(f.name)
      +'<span class="fc-x" onclick="event.stopPropagation();removeFile('+i+')">&times;</span></span>';
  }).join('');
  var n=SF.length;
  document.getElementById('fcount').textContent=
    n>0?n+' file'+(n!==1?'s':'')+' ready — click ▶ Analyze PDFs':'';
  document.getElementById('btn-analyze').className=
    'btn '+(n>0?'btn-green':'btn-blue');
}
function removeFile(i){SF.splice(i,1);renderFileList();}
function dov(e){e.preventDefault();e.currentTarget.classList.add('drag');}
function dlv(e){e.currentTarget.classList.remove('drag');}
function ddp(e){e.preventDefault();e.currentTarget.classList.remove('drag');onF(e.dataTransfer.files);}

/* ─── upload & analyze ─── */
async function doUpload(){
  if(!SF.length){showUploadErr('Please choose at least one PDF file first.');return;}
  showPage('progress');
  setStep('upload','active');
  var fd=new FormData();
  SF.forEach(function(f){fd.append('pdfs',f);});
  try{
    var r=await fetch('/api/upload',{method:'POST',body:fd});
    var d=await r.json();
    if(d.error){showPage('upload');showUploadErr(d.error);return;}
    setStep('upload','done'); setStep('ocr','active');
    poll(d.job_id);
  }catch(err){
    showPage('upload');showUploadErr('Could not reach server: '+err.message);
  }
}

async function doSample(){
  showPage('progress');
  setStep('upload','done'); setStep('ocr','active');
  try{
    var r=await fetch('/api/sample');
    var d=await r.json();
    if(d.error){showPage('upload');showUploadErr(d.error);return;}
    poll(d.job_id);
  }catch(err){
    showPage('upload');showUploadErr('Could not reach server: '+err.message);
  }
}

/* ─── polling ─── */
function setStep(id,state){
  var el=document.getElementById('step-'+id);
  if(!el)return;
  el.className='step '+(state||'');
}
function poll(jid){
  if(PT)clearInterval(PT);
  PT=setInterval(async function(){
    try{
      var r=await fetch('/api/status/'+jid),d=await r.json();
      var msg=(d.message||'').toLowerCase();
      document.getElementById('pmsg').textContent=d.message||'…';
      document.getElementById('pfill').style.width=(d.progress||0)+'%';
      if(msg.indexOf('ocr')>=0||msg.indexOf('textract')>=0){
        setStep('upload','done');setStep('ocr','active');
      } else if(msg.indexOf('pars')>=0){
        setStep('ocr','done');setStep('parse','active');
      } else if(msg.indexOf('cluster')>=0){
        setStep('parse','done');setStep('cluster','active');
      } else if(d.status==='done'){
        setStep('cluster','done');
      }
      if(d.status==='done'){
        clearInterval(PT);DATA=d;
        setTimeout(function(){renderResults();},300);
      } else if(d.status==='error'){
        clearInterval(PT);showPage('upload');showUploadErr('Error: '+d.message);
      }
    }catch(err){
      clearInterval(PT);showPage('upload');showUploadErr('Lost connection: '+err.message);
    }
  },1500);
}

/* ─── results ─── */
function renderResults(){
  if(!DATA||!DATA.questions||!DATA.questions.length){
    showPage('upload');
    showUploadErr('No questions found in these PDFs.');
    return;
  }
  showPage('results');
  document.getElementById('hdr-info').textContent=
    DATA.pdfs.length+' PDF'+(DATA.pdfs.length!==1?'s':'')+' · '+DATA.questions.length+' Pages';
  document.getElementById('qlist').innerHTML=DATA.questions.map(function(q,i){
    var pgNum=q.question.replace(/[^0-9]/g,'')||''+(i+1);
    return'<button class="qbtn'+(i===0?' on':'')+'" onclick="selQ('+i+')">'
      +'<span class="qnum">'+pgNum+'</span>'
      +esc(q.question)+'<span class="qpill">'+q.n_submissions+'</span></button>';
  }).join('');
  selQ(0);
}
function selQ(i){
  CQ=i;
  document.querySelectorAll('.qbtn').forEach(function(b,j){b.classList.toggle('on',j===i);});
  renderQ(DATA.questions[i]);
}
function renderQ(q){
  var nGroups=q.clusters.length;
  document.getElementById('res-main').innerHTML=
    '<div class="qhdr"><div class="qtitle">'+esc(q.question)+'</div>'
    +'<span class="pill pill-b">'+q.n_submissions+' sub.</span>'
    +'<span class="pill pill-b">'+nGroups+' unique answer'+(nGroups!==1?'s':'')+'</span></div>'
    +buildGroups(q);
}
function buildGroups(q){
  return q.clusters.map(function(c){
    var ans=c.answer||c.members[0].preview||'[blank]';
    var n=c.members.length;
    return'<div class="ccard">'
      +'<div class="chead">'
      +'<div class="dot" style="background:'+c.color+'"></div>'
      +'<b>Group '+c.id+'</b>'
      +'<span class="csub">&nbsp;&mdash; '+n+' student'+(n!==1?'s':'')+'</span>'
      +'<div class="cact"><button class="apply-btn" onclick="applyC(this,'+c.id+',&quot;'+q.question+'&quot;,'+n+')">Apply to all '+n+'</button></div>'
      +'</div>'
      +'<div class="ans-box"><div class="ans-label">Final Answer</div><div class="ans-body">'+esc(ans)+'</div></div>'
      +'<div class="stu-list"><div class="stu-label">Students ('+n+')</div>'
      +c.members.map(function(m){return stuCard(m,c.color);}).join('')
      +'</div>'
      +'</div>';
  }).join('');
}
function stuCard(m,color){
  var img=m.image_b64
    ?'<img class="mimg" src="data:image/png;base64,'+m.image_b64+'" onclick="olb(this.src)" title="Click to enlarge"/>'
    :'';
  return'<div class="stu-card">'
    +(img?'<div class="stu-thumb">'+img+'</div>':'')
    +'<div class="stu-name">'+esc(m.id)+'</div>'
    +'</div>';
}
function applyC(btn,cid,ql,n){btn.textContent='Applied to '+n+'!';btn.classList.add('done');btn.disabled=true;}
function showUploadErr(msg){
  document.getElementById('upload-err').innerHTML='<div class="err-box">&#9888; '+esc(msg)+'</div>';
}
function olb(src){document.getElementById('lbimg').src=src;document.getElementById('lb').classList.add('open');}
function clb(){document.getElementById('lb').classList.remove('open');}
document.addEventListener('keydown',function(e){if(e.key==='Escape')clb();});
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
</script>
</body>
</html>"""


if __name__ == "__main__":
    missing = [v for v in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            "Missing AWS environment variables: " + ", ".join(missing) + "\n"
            "  export AWS_DEFAULT_REGION='us-west-2'\n"
            "  export AWS_ACCESS_KEY_ID='...'\n"
            "  export AWS_SECRET_ACCESS_KEY='...'\n"
            "  export AWS_SESSION_TOKEN='...'   # for temporary credentials"
        )
    print("\n  ClusterGrade AI - PDF Analysis Tool")
    print("   Open in browser: http://localhost:8080")
    print("   Ready to analyze exam submissions!\n")
    app.run(debug=False, port=8080, threaded=True)
