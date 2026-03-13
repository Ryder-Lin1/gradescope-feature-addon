# ClusterGrade AI — Devpost Story

## Inspiration

Every semester, teaching assistants at large universities face the same grind: reading hundreds of near-identical exam answers, writing the same feedback comment over and over, and trying to stay consistent across a mountain of submissions. In a single 500-student introductory CS course, a TA can spend **20+ hours** grading one midterm — and the vast majority of that time is spent writing the *same three comments* about the *same recurring mistake*.

We asked ourselves: **what if the grading system could recognize that 47 students all wrote the same wrong answer, and let the TA grade them all at once?**

Existing tools don't solve this. Plagiarism detectors like MOSS find *copied* work, not *independently identical* answers. AI auto-graders (GPT-based) grade each submission in isolation, producing inconsistent feedback. Gradescope's rubric builder helps, but the TA still has to *read every single answer* to decide which rubric item applies.

No tool closes the loop between **grouping identical answers** and **applying feedback at scale**. That's what ClusterGrade AI does.

---

## What It Does

ClusterGrade AI is a web-based tool that:

1. **Accepts student exam PDFs** — one PDF per student, each page representing one question
2. **OCRs the answer boxes** using **AWS Textract** — extracting only the final answer written inside the designated box on each page
3. **Groups identical answers together** — students who wrote the same final answer are placed in the same group
4. **Displays results in a clean dashboard** — the TA sees each unique answer (in **bold red uppercase**) with the list of students who gave that answer, along with thumbnail previews of their actual handwriting
5. **Enables one-click batch grading** — the TA reviews one answer, clicks "Apply to all $n$", and the grade propagates to every student in that group

The key insight is that for many exam questions — especially in math, physics, and introductory CS — the final answer is either right or wrong. If 30 students all wrote $\frac{3}{4}$ in the box, the TA should only have to grade that answer **once**.

---

## How We Built It

### Architecture

The entire application is a single-file **Flask** server (`app.py`, ~940 lines) with an embedded HTML/CSS/JS frontend — no build tools, no npm, no webpack. Just Python and a browser.

```
Student PDFs
    │
    ▼
pdf2image (PDF → PNG, one per page)
    │
    ▼
AWS Textract  (detect_document_text API)
    │  returns LINE blocks with bounding-box geometry
    ▼
extract_boxed_answer()
    │  uses indentation + vertical gap heuristics
    │  to separate question text from the boxed answer
    ▼
normalize_answer()
    │  lowercase, strip whitespace, remove punctuation
    ▼
group_by_answer()
    │  exact-match grouping by normalized answer text
    ▼
JSON API → Frontend Dashboard
```

### Key Technical Decisions

**Box Detection via Geometry Heuristics.** AWS Textract returns LINE-level bounding boxes with coordinates $(x, y, w, h)$ normalized to $[0, 1]$. We observed that answer boxes on exam pages are typically indented from the left margin. Our heuristic:

$$
\text{is\_boxed}(line) = \begin{cases} \text{true} & \text{if } line.\text{left} > \min(\text{lefts}) + 0.03 \\ \text{false} & \text{otherwise} \end{cases}
$$

If no indentation is detected, we fall back to splitting at the largest vertical gap:

$$
\text{gap}_i = line_{i+1}.\text{top} - (line_i.\text{top} + line_i.\text{height})
$$

and split at $\arg\max_i(\text{gap}_i)$ when $\max(\text{gap}_i) > 0.015$.

**Exact-Match Grouping over ML Clustering.** We initially built a full TF-IDF + K-Means clustering pipeline with silhouette score optimization:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. But we realized that for boxed final answers, **exact match is both simpler and more correct**. A student who wrote "$x = 5$" and a student who wrote "$x = 4$" should *never* be in the same group, even if TF-IDF thinks they're similar. We replaced the entire ML pipeline with a dictionary-based grouping:

```python
groups = defaultdict(list)
for sid, answer in submissions:
    key = normalize_answer(answer)  # lowercase, strip, remove punctuation
    groups[key].append(sid)
```

This runs in $O(n)$ time versus $O(n \cdot k \cdot d)$ for K-Means, and produces perfect groupings every time.

**Bypassing Jinja2.** Flask's `render_template_string()` uses Jinja2 syntax (`{{ }}`), which collides with JavaScript object literals and CSS custom properties. Rather than escaping hundreds of braces, we serve the HTML directly:

```python
return Response(HTML, mimetype="text/html")
```

**CSS Specificity Wars.** Our three-page navigation (`upload → progress → results`) toggles pages via `.active` class. But `#page-upload { display: flex }` always beat `.page { display: none }` because ID selectors have higher specificity ($S_{ID} = 100$ vs $S_{class} = 10$). The fix:

```css
.page { display: none !important }
#page-upload.active { display: flex !important }
```

### Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3.13, Flask, flask-cors |
| OCR | AWS Textract (`detect_document_text`) |
| PDF Processing | `pdf2image` + Poppler |
| Frontend | Vanilla HTML/CSS/JS (embedded in Python) |
| Answer Grouping | Exact-match with normalization |
| Deployment | Single-file server, port 8080 |

---

## Challenges We Faced

### 1. macOS Port Conflicts
Port 5000 is used by AirPlay Receiver on modern macOS. We discovered this the hard way when Flask silently started but the browser showed Apple's AirPlay page. Moved to 5001, then finally to **8080**.

### 2. AWS Textract Geometry Interpretation
Textract returns bounding boxes as fractions of page dimensions, not pixels. Understanding that `Left: 0.15` means "15% from the left edge of the page" — and using that to detect indented answer boxes — required careful experimentation with real exam scans.

### 3. The Jinja2 / JavaScript Brace Collision
Our first version used `render_template_string()`, which silently ate every `{}` in our JavaScript and CSS. Functions like `const obj = {key: val}` became `const obj =` (empty). CSS variables like `var(--blue)` became `var()`. The error was invisible because Jinja2 doesn't throw on unknown variables — it just renders them as empty strings. Debugging this took longer than writing the feature.

### 4. Escaped Quotes in Inline Event Handlers
Python triple-quoted strings containing JS `onclick` handlers with single quotes (`onclick="fn('arg')"`) produced backslash-escaped quotes (`onclick="fn(\'arg\')"`) in the HTML. Browsers rejected these silently. The fix was using HTML entities: `&quot;` instead of escaped quotes.

### 5. From Clustering to Exact Match
Our original ML pipeline (TF-IDF vectorization → optimal $k$ search → K-Means → PCA visualization → cosine similarity heatmap) was technically impressive but **wrong for the use case**. Two answers that differ by a single digit are *completely different answers* to a math problem, but TF-IDF considers them nearly identical. Ripping out 200+ lines of ML code and replacing them with a 10-line dictionary grouping was humbling but correct.

### 6. CSS Specificity Overriding Page Visibility
Pages that should have been hidden stayed visible because `#page-results { display: flex }` (specificity 100) always overrode `.page { display: none }` (specificity 10). The `!important` fix felt inelegant but was the most reliable solution for a single-file embedded stylesheet.

---

## What We Learned

- **Simpler is often better.** A 10-line exact-match grouping replaced a 200-line ML pipeline and produced better results. Not every problem needs machine learning.
- **AWS Textract's LINE geometry is powerful.** Beyond just OCR text, the bounding box coordinates let you understand *where* text is on a page — enabling answer-box detection without any computer vision model.
- **Single-file apps have tradeoffs.** Embedding HTML/CSS/JS inside a Python string avoids build complexity but creates novel escaping challenges (Jinja2, quote escaping, brace collisions).
- **CSS specificity matters.** In a complex single-page app, understanding the cascade is not optional. ID selectors will silently override class-based toggles.
- **Test with real data early.** Our clustering pipeline looked great on synthetic data but fell apart on real handwritten exam answers where "5" and "6" are not "similar" — they're completely different grades.

---

## What's Next

- **Amazon Bedrock Integration** — Send one representative answer per group + the professor's rubric to Claude/Titan to auto-generate deduction amounts and feedback comments
- **Gradescope API Integration** — Push approved grades directly back into Gradescope via their REST API, closing the loop entirely
- **Partial Credit Clustering** — For free-response questions where exact match isn't sufficient, re-enable the TF-IDF/K-Means pipeline as a configurable option
- **Multi-Question Rubric View** — Show a matrix of students × questions with group colors, so TAs can spot patterns across an entire exam at a glance

---

## Built With

`python` · `flask` · `aws-textract` · `boto3` · `pdf2image` · `scikit-learn` · `html` · `css` · `javascript`
