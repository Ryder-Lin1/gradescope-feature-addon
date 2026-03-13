"""
generate_samples.py
-------------------
Generates 10 synthetic "student answer" images in the sample_answers/ folder.
Each image simulates a short handwritten-style text answer to a CS question.
The answers intentionally fall into 3 error/pattern groups so the clustering
algorithm has meaningful signal to work with.

No external fonts required — uses Pillow's built-in default font, with a
slight random rotation to mimic real scanned handwriting.

Run:
    python generate_samples.py
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "sample_answers"

# ---------------------------------------------------------------------------
# Synthetic answers — 3 intentional clusters:
#   Cluster A: missing base case / infinite recursion errors  (4 answers)
#   Cluster B: off-by-one / loop bound errors                 (3 answers)
#   Cluster C: correct answers with minor wording variation   (3 answers)
# ---------------------------------------------------------------------------
ANSWERS = [
    # Cluster A – missing base case
    (
        "answer_02.png",
        [
            "The base case is missing so the",
            "recursion never terminates.",
            "The function keeps calling itself",
            "infinitely without a stopping",
            "condition, causing a stack overflow.",
        ],
    ),
    (
        "answer_05.png",
        [
            "There is no base case defined.",
            "The recursive call repeats forever",
            "because the function never reaches",
            "a terminating condition.",
            "This causes infinite recursion.",
        ],
    ),
    (
        "answer_07.png",
        [
            "Missing base case means recursion",
            "never stops. The stack overflows",
            "because the stopping condition was",
            "never written into the function.",
            "Infinite loop of recursive calls.",
        ],
    ),
    (
        "answer_09.png",
        [
            "The recursive function lacks a",
            "base case. Without it the function",
            "calls itself infinitely and the",
            "program crashes with a stack",
            "overflow error.",
        ],
    ),
    # Cluster B – off-by-one
    (
        "answer_01.png",
        [
            "The loop runs n+1 times instead",
            "of n, causing an off-by-one error.",
            "The index goes out of bounds on",
            "the final iteration because the",
            "condition should be i < n not i <= n.",
        ],
    ),
    (
        "answer_04.png",
        [
            "Off-by-one: loop bound uses <=",
            "instead of <. This makes the loop",
            "execute one extra time and access",
            "an index that is out of range.",
            "Change i <= n to i < n.",
        ],
    ),
    (
        "answer_08.png",
        [
            "The loop condition i <= n should",
            "be i < n. As written, the loop",
            "iterates one time too many and",
            "reads past the end of the array,",
            "causing an index out of bounds.",
        ],
    ),
    # Cluster C – correct answers
    (
        "answer_03.png",
        [
            "The base case returns 0 when n",
            "equals 0. The recursive step",
            "reduces n by 1 each call.",
            "The function correctly computes",
            "the sum from 0 to n.",
        ],
    ),
    (
        "answer_06.png",
        [
            "Correctly identifies the base",
            "case: return 0 when n == 0.",
            "Each recursive call decrements n",
            "so the function terminates and",
            "returns the correct cumulative sum.",
        ],
    ),
    (
        "answer_10.png",
        [
            "Base case n == 0 returns 0.",
            "Recursive case returns n plus",
            "f(n-1) which is correct.",
            "The function terminates because",
            "n decreases toward the base case.",
        ],
    ),
]


def make_image(lines: list[str], filename: str, output_dir: str) -> None:
    """Render a list of text lines onto a white image and save it."""
    width, height = 480, 200
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use Pillow's default font (always available, no installation needed)
    try:
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    y = 20
    for line in lines:
        # Add subtle jitter to x position to simulate handwriting
        x = 20 + random.randint(-3, 3)
        draw.text((x, y), line, fill=(10, 10, 10), font=font)
        y += 28

    # Slight random rotation to mimic a scanned page
    angle = random.uniform(-1.5, 1.5)
    img = img.rotate(angle, fillcolor=(255, 255, 255))

    path = os.path.join(output_dir, filename)
    img.save(path)
    print(f"  Created: {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating {len(ANSWERS)} sample answer images in '{OUTPUT_DIR}/'...\n")
    for filename, lines in ANSWERS:
        make_image(lines, filename, OUTPUT_DIR)
    print(f"\nDone. {len(ANSWERS)} images written to '{OUTPUT_DIR}/'.")
    print("Now run:  python cluster_prototype.py --input_dir sample_answers/ --n_clusters 3")


if __name__ == "__main__":
    random.seed(42)
    main()
