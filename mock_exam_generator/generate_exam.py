#!/usr/bin/env python3
"""
Mock Final Exam Generator for Statistical Machine Learning.

Reads course materials (lectures, midterm 1, study guide) and produces:
  1. A predicted-questions mock final exam
  2. A separate answer key with detailed solutions
  3. High-yield topic predictions calibrated from the materials

Usage:
    python generate_exam.py materials/                      # all files in dir
    python generate_exam.py lec1.pdf lec2.pdf midterm.pdf   # specific files
    python generate_exam.py -o exam.md materials/

Requires:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
"""

import argparse
import base64
import os
import sys
from pathlib import Path

import anthropic


SUPPORTED_DOC_EXTS = {".pdf"}
SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
SUPPORTED_EXTS = SUPPORTED_DOC_EXTS | SUPPORTED_TEXT_EXTS | SUPPORTED_IMAGE_EXTS

SPLIT_MARKER = "---ANSWER_KEY_BELOW---"

SYSTEM_PROMPT = f"""You are an expert in Statistical Machine Learning and exam design. Your job is to analyze a student's course materials (lecture notes, prior exams, study guides) and produce a high-quality mock final exam that closely mirrors what is likely to appear on the actual final.

Your analysis should:
1. **Identify high-yield topics** — what gets the most lecture time, what appeared on the midterm, what the study guide emphasizes. Cumulative finals usually re-test midterm topics with twists, plus everything since.
2. **Match the instructor's style** — calibrate from the midterm to predict format, difficulty, notation conventions, and the computational-vs-conceptual mix.
3. **Sample broadly** — cover all major topics, weighted by likely importance.
4. **Predict trick areas** — concepts commonly tested with subtle gotchas: bias-variance, regularization tradeoffs, kernel tricks, EM convergence, MLE vs MAP, identifiability, generalization bounds, overfitting diagnostics, etc.

Output format (Markdown). Follow the structure exactly:

# Mock Final Exam: Statistical Machine Learning

## Predicted High-Yield Topics
[Bulleted list of 6-10 topics most likely to appear, each with a brief justification grounded in the materials — cite where the topic was emphasized (e.g. "Lecture 5 spent ~30 min on this; midterm Q3 tested the basic case").]

## Exam

### Part I: Short Answer / Conceptual ([N] questions)
### Part II: Derivations / Proofs ([N] questions)
### Part III: Computational / Numerical ([N] questions)
### Part IV: Long-Form / Application ([N] questions)

[Format each question clearly with point values. Match the midterm's style for question count and difficulty distribution. If the midterm had ~6 questions over 90 minutes, scale appropriately for a typical 3-hour final.]

{SPLIT_MARKER}

# Answer Key

[Detailed solutions for every question. Show derivations step-by-step. For conceptual questions, give the ideal answer plus 1-2 common wrong answers students give and why they're wrong.]

# Study Strategy
[Brief: what to drill before the exam, in priority order, based on what's most likely to appear. Be specific — name the topics, not generic advice.]

Be rigorous, calibrated, and specific to THIS course's materials — not a generic SML exam. Use the same notation as the materials (e.g. if they use $\\hat\\beta$ for OLS coefficients, you do too). Quote the materials when justifying predictions."""


def load_file_block(path: Path) -> list[dict]:
    """Return content blocks for one file (label + document/image, or just labelled text)."""
    suffix = path.suffix.lower()
    label = {"type": "text", "text": f"\n\n=== File: {path.name} ===\n"}

    if suffix in SUPPORTED_DOC_EXTS:
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return [
            label,
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                },
                "title": path.stem,
            },
        ]
    if suffix in SUPPORTED_TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
        label["text"] += text
        return [label]
    if suffix in SUPPORTED_IMAGE_EXTS:
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        ext = suffix.lstrip(".")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return [
            label,
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            },
        ]
    raise ValueError(f"Unsupported file type: {path}")


def collect_files(inputs: list[str]) -> list[Path]:
    """Expand directories and validate paths."""
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
                    files.append(child)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTS:
                raise ValueError(f"Unsupported file: {p}")
            files.append(p)
        else:
            raise FileNotFoundError(f"Not found: {inp}")
    if not files:
        raise ValueError(
            f"No supported files found. Supported: {', '.join(sorted(SUPPORTED_EXTS))}"
        )
    return files


def build_user_content(files: list[Path]) -> list[dict]:
    """User message content: intro + materials (cached) + generation prompt."""
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Below are {len(files)} file(s) of course materials for a "
                "Statistical Machine Learning course. They include lecture notes, "
                "midterm 1 (with solutions if available), and a study guide. "
                "File names indicate which is which."
            ),
        }
    ]
    for f in files:
        content.extend(load_file_block(f))

    # Cache the materials prefix; the generate-instruction below is volatile.
    content[-1]["cache_control"] = {"type": "ephemeral"}

    content.append(
        {
            "type": "text",
            "text": (
                "\n\nGenerate the mock final exam now. Follow the format from the "
                "system prompt exactly. Calibrate question count, point distribution, "
                "and difficulty to match the midterm. Use the same notation."
            ),
        }
    )
    return content


def split_and_write(full_output: str, output_path: Path) -> None:
    """Split exam from answer key; write two files if marker present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if SPLIT_MARKER in full_output:
        exam, answers = full_output.split(SPLIT_MARKER, 1)
        answer_path = output_path.with_stem(output_path.stem + "_answers")
        output_path.write_text(exam.strip() + "\n")
        answer_path.write_text(answers.strip() + "\n")
        print(f"\n\nExam:    {output_path}", file=sys.stderr)
        print(f"Answers: {answer_path}", file=sys.stderr)
    else:
        output_path.write_text(full_output)
        print(f"\n\nSaved to: {output_path}", file=sys.stderr)


def generate(files: list[Path], output_path: Path) -> None:
    client = anthropic.Anthropic()

    print(f"Materials ({len(files)} file(s)):", file=sys.stderr)
    total_kb = 0.0
    for f in files:
        kb = f.stat().st_size / 1024
        total_kb += kb
        print(f"  {f.name}  ({kb:.1f} KB)", file=sys.stderr)
    print(f"Total: {total_kb:.1f} KB\n", file=sys.stderr)

    user_content = build_user_content(files)

    print("Generating exam (streaming)...\n", file=sys.stderr)

    chunks: list[str] = []
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            print(text, end="", flush=True)
        final = stream.get_final_message()

    full_output = "".join(chunks)
    split_and_write(full_output, output_path)

    u = final.usage
    print(f"\nUsage: input={u.input_tokens}  output={u.output_tokens}", file=sys.stderr)
    if u.cache_creation_input_tokens:
        print(f"  cache write: {u.cache_creation_input_tokens}", file=sys.stderr)
    if u.cache_read_input_tokens:
        print(f"  cache read:  {u.cache_read_input_tokens}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a Statistical Machine Learning mock final exam.",
        epilog="Tip: name files clearly (lecture_01.pdf, midterm1.pdf, study_guide.pdf).",
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Files or directories with course materials (PDF/TXT/MD/PNG/JPG)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/mock_final.md"),
        help="Output path (default: output/mock_final.md). Answers go to a sibling _answers.md file.",
    )
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set in environment.")

    try:
        files = collect_files(args.inputs)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    generate(files, args.output)


if __name__ == "__main__":
    main()
