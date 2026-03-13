# ClusterGrade AI
### A Hybrid Computer Vision + Generative AI Grading System
**Submission for the UofT "Enhancing Teaching Efficiency with Intelligent Course Support" Challenge**

---

## 1. The Value Proposition

### The Problem
Instructors and teaching staff at large universities spend a disproportionate amount of time managing **routine, repetitive course tasks** — primarily:
- Responding to the same student questions over and over
- Developing and refining grading rubrics from scratch each term
- Individually grading hundreds of near-identical submissions, writing the same feedback comment dozens of times

In a 500-student introductory CS course, a TA may spend 20+ hours grading a single midterm — the vast majority of that time writing the **same three comments** about the same base-case error.

### The Flaw in Current Solutions
| Approach | Limitation |
|---|---|
| Clustering-only tools (e.g., MOSS) | Groups similar submissions, but still requires manual grading of every individual |
| AI-only grading (e.g., GPT-based auto-graders) | Grades each submission independently → **inconsistent feedback** across the class |
| Gradescope rubric builder | Reduces some repetition, but the TA still reads every answer individually to apply rubric items |

No current solution closes the loop between *grouping similar errors* and *generating consistent, rubric-aligned feedback at scale*.

### Our Solution: ClusterGrade AI
A hybrid AWS architecture that uses **Computer Vision** to group submissions with identical logic errors, and **Amazon Bedrock** to generate high-quality, rubric-aligned feedback for the entire group at once.

> **One TA review. Consistent feedback. Applied to 50 students in a single click.**

---

## 2. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ClusterGrade AI                          │
│                                                                 │
│  [Scanned PDFs / Assignment Images]                             │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │ Amazon Textract │  ← OCR: extracts handwritten text & code  │
│  └────────┬────────┘                                            │
│           │  raw text strings per submission                    │
│           ▼                                                     │
│  ┌──────────────────────────────┐                               │
│  │  Python + Scikit-Learn       │                               │
│  │  • TF-IDF / Sentence embeddings                             │
│  │  • K-Means clustering        │  ← groups identical errors   │
│  └──────────────┬───────────────┘                               │
│                 │  cluster representatives                       │
│                 ▼                                               │
│  ┌─────────────────────────────┐                                │
│  │      Amazon Bedrock         │  ← LLM + professor's rubric   │
│  │  (e.g., Claude / Titan)     │    generates deduction +      │
│  │                             │    personalized feedback       │
│  └──────────────┬──────────────┘                                │
│                 │                                               │
│                 ▼                                               │
│  ┌─────────────────────────────┐                                │
│  │  Human-in-the-Loop Approval │  ← TA reviews feedback ONCE   │
│  │  (TA Dashboard)             │    → applied to entire cluster │
│  └─────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Technology | Role |
|---|---|---|
| **Ingestion & OCR** | Amazon Textract | Extracts handwritten text and code from scanned midterm PDFs or assignment images |
| **Vectorization** | Python + `sentence-transformers` | Converts extracted text into semantic embeddings that capture meaning, not just keywords |
| **Clustering** | Scikit-Learn K-Means | Groups submissions that took the exact same approach or made the exact same logic error |
| **Generative Assessment** | Amazon Bedrock (Claude / Titan) | Sends one representative answer per cluster + the professor's rubric → generates deduction and feedback |
| **Approval & Distribution** | TA Web Dashboard | TA reviews AI-generated feedback once; upon approval, the grade is applied to all N students in that cluster |

---

## 3. Pilot Program

**Target Course:** Foundational Computer Science courses (e.g., CSC148 / CSC207 at UofT)

**Why this is the perfect testbed:**
- Hundreds of students write nearly identical recursive functions, memory model diagrams, and logic proofs
- Common error patterns (missing base case, off-by-one in loop bounds, incorrect pointer assignment) recur across almost every student cohort
- Anonymized past midterms are readily available for safe, IRB-compliant testing

**Pilot Phases:**

| Phase | Description | Success Metric |
|---|---|---|
| **Phase 1 – Clustering Validation** | Run 3 years of anonymized CSC148 midterms through the clustering pipeline (no AI grading). Manually verify that clusters correspond to genuine shared errors. | ≥80% cluster purity (same error type per cluster) |
| **Phase 2 – AI Feedback Quality** | Send cluster representatives + rubric to Bedrock. Have 3 TAs blind-rate AI feedback vs. human feedback on a 5-point rubric-alignment scale. | AI feedback rated ≥4/5 on consistency and accuracy |
| **Phase 3 – Live TA Pilot** | Deploy to one live section (~200 students). Measure TA grading time vs. historical baseline. | ≥40% reduction in time-to-grade |

---

## 4. Expected Outcomes & Business Impact

| Challenge Criterion | ClusterGrade AI Impact |
|---|---|
| **Reduced Administrative Burden** | By grading clusters instead of individuals, TA grading time is cut by an estimated **40–60%**. A TA who previously spent 20 hours on a midterm now spends 6–8 hours. |
| **Faculty Productivity** | Streamlined workflows free instructors from repetitive feedback writing, allowing them to focus on high-value teaching and mentoring interactions. |
| **Scalable Course Management** | A single TA can efficiently manage feedback for a 500-student lecture section. Adding more students does **not** linearly increase grading overhead — cluster sizes grow, not cluster counts. |
| **Consistent Student Experience** | Every student in a cluster receives the same high-quality, rubric-aligned feedback. No more lottery-of-which-TA-graded-you inconsistency. |
| **Data-Driven Curriculum Improvement** | Cluster analytics reveal the most common misconceptions across a cohort, giving instructors actionable data to improve future lectures and assignment design. |

---

## 5. Local Prototype

A self-contained Python prototype is included to demonstrate the core clustering pipeline **without any cloud dependencies**.

### What it does
1. Reads a folder of images containing handwritten or typed text answers
2. Extracts text using **Tesseract OCR** (local, free alternative to Amazon Textract)
3. Converts each answer to a **TF-IDF vector**
4. Runs **K-Means clustering** to group similar answers
5. Prints a cluster report showing which submissions share the same error pattern

### Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# (macOS) Install Tesseract engine
brew install tesseract

# Generate sample test images (10 synthetic handwritten-style answers)
python generate_samples.py

# Run the clustering pipeline
python cluster_prototype.py --input_dir sample_answers/ --n_clusters 3
```

### Sample Output

```
============================================================
  ClusterGrade AI — Local Clustering Prototype
============================================================
Processed 10 submissions.
Optimal cluster range tested: 2 – 5

CLUSTER 1  (4 submissions)
  Representative answer: "answer_02.png"
  Preview: "the base case is missing so the recursion never terminates..."
  Members: answer_02.png, answer_05.png, answer_07.png, answer_09.png

CLUSTER 2  (3 submissions)
  Representative answer: "answer_01.png"
  Preview: "the loop runs n+1 times instead of n causing an off by one..."
  Members: answer_01.png, answer_04.png, answer_08.png

CLUSTER 3  (3 submissions)
  Representative answer: "answer_03.png"
  Preview: "correctly identifies the base case and returns 0 when n equals..."
  Members: answer_03.png, answer_06.png, answer_10.png

Silhouette Score: 0.72  (higher = more distinct clusters)
============================================================
```

---

## Repository Structure

```
gradescope-feature-addon/
├── README.md                  # This proposal
├── requirements.txt           # Python dependencies
├── generate_samples.py        # Generates 10 synthetic test answer images
├── cluster_prototype.py       # Main OCR + clustering pipeline
└── sample_answers/            # Auto-generated by generate_samples.py
```

---

## Why ClusterGrade AI Wins

- ✅ **Directly addresses the challenge area** — reduces instructor time on routine grading tasks
- ✅ **Hybrid approach** — combines proven CV/OCR with cutting-edge generative AI, avoiding the pitfalls of either alone
- ✅ **Human-in-the-loop** — AI never grades without TA approval, maintaining academic integrity
- ✅ **Scalable by design** — AWS-native architecture grows with enrollment, not against it
- ✅ **Provable with a prototype** — the local pipeline can be demonstrated live to judges today
