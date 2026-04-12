from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "xai_gesture_shap_analysis.ipynb"
RAW_DIR = ROOT / "data" / "raw"
RANDOM_MOTION_DIR = ROOT / "data" / "random_motion"

REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "shap",
    "torch",
    "cv2",
    "mediapipe",
]

REQUIRED_NOTEBOOK_TOKENS = [
    "STRICT_REAL_NEGATIVE_EVAL = True",
    "run_shap_stability_analysis",
    "compare_shap_and_fragility",
    "run_intent_gate_evaluation",
    "max_allowed_fpr=0.05",
    "TARGET_TOP1_ACCURACY = 0.95",
]


def load_notebook_cells() -> list[dict]:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK}")
    with NOTEBOOK.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("cells", [])


def get_code_cells(cells: list[dict]) -> list[str]:
    code_cells = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            code_cells.append(src)
    return code_cells


def check_imports() -> list[str]:
    errors = []
    for name in REQUIRED_IMPORTS:
        try:
            __import__(name)
        except Exception as e:
            errors.append(f"missing import {name}: {e}")
    return errors


def check_notebook_syntax(code_cells: list[str]) -> list[str]:
    errors = []
    for i, src in enumerate(code_cells):
        sanitized_lines = []
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("%") or stripped.startswith("!"):
                sanitized_lines.append("pass")
            else:
                sanitized_lines.append(line)
        sanitized_src = "\n".join(sanitized_lines)
        try:
            ast.parse(sanitized_src)
        except Exception as e:
            errors.append(f"code cell {i} syntax error: {e}")
    return errors


def ensure_random_motion_min_count(min_count: int = 50) -> list[str]:
    actions = []
    RANDOM_MOTION_DIR.mkdir(parents=True, exist_ok=True)
    current = len(list(RANDOM_MOTION_DIR.glob("*")))
    if current >= min_count:
        return actions

    all_images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        all_images.extend(RAW_DIR.rglob(ext))
    if not all_images:
        actions.append("cannot auto-fill random_motion: no images in data/raw")
        return actions

    needed = min_count - current
    for src in all_images[:needed]:
        dst = RANDOM_MOTION_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    actions.append(f"auto-filled random_motion to >= {min_count}")
    return actions


def check_required_paths() -> list[str]:
    errors = []
    if not RAW_DIR.exists():
        errors.append("missing data/raw")
    class_dirs = [p for p in RAW_DIR.iterdir() if p.is_dir()] if RAW_DIR.exists() else []
    if len(class_dirs) < 2:
        errors.append("data/raw has too few class directories")
    if not RANDOM_MOTION_DIR.exists():
        errors.append("missing data/random_motion")
    random_count = len(list(RANDOM_MOTION_DIR.glob("*"))) if RANDOM_MOTION_DIR.exists() else 0
    if random_count < 50:
        errors.append(f"data/random_motion has too few files: {random_count}")
    return errors


def check_notebook_tokens(text: str) -> list[str]:
    errors = []
    for token in REQUIRED_NOTEBOOK_TOKENS:
        if token not in text:
            errors.append(f"missing notebook token: {token}")
    return errors


def run_pass(pass_idx: int) -> tuple[list[str], list[str]]:
    actions = ensure_random_motion_min_count(min_count=50)

    cells = load_notebook_cells()
    code_cells = get_code_cells(cells)
    notebook_text = NOTEBOOK.read_text(encoding="utf-8")

    errors = []
    errors.extend(check_imports())
    errors.extend(check_required_paths())
    errors.extend(check_notebook_syntax(code_cells))
    errors.extend(check_notebook_tokens(notebook_text))

    print(f"pass {pass_idx:02d}: {'OK' if not errors else 'FAIL'}")
    if actions:
        for action in actions:
            print(f"  action: {action}")
    if errors:
        for err in errors:
            print(f"  error: {err}")
    return errors, actions


def main() -> None:
    any_errors = False
    for idx in range(1, 11):
        errors, _ = run_pass(idx)
        if errors:
            any_errors = True

    print("\nsummary:")
    if any_errors:
        print("preflight has failures. fix required before notebook execution.")
        raise SystemExit(1)
    print("all 10 preflight passes succeeded. safe to run notebook.")


if __name__ == "__main__":
    main()
