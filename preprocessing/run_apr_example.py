"""LADAR execute-APR demo on a bundled case package.

Runs the two-stage execute-phase repair (Stage 1 error analysis + Stage 2
context-based repair) against the provided ORIGIN PA log and the
original malicious source, validates the patch with ``compile()``, and
writes the patched file to
``data/example_package/<pkg>-<ver>/patched_runtime/``.

This script does not rerun the OpenSSF Package Analysis tool on the
patched output; it stops after producing a compile-valid patch. The
reference patched source trees under
``data/example_package/<pkg>-<ver>/patched/`` and their companion
``data/pa_logs/*_ladar_execute.json`` PA logs show what the paper's
full pipeline produces when the PA sandbox is also rerun.

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...

Run:
    python preprocessing/run_apr_example.py [safepackage|equests|r4quests]
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocessing.execute_apr import (
    ExecuteStageErrorMessageProcessor,
    Stage1PromptGenerator,
    CodeContextExtractor,
    Stage2PromptGenerator,
    Postprocessor,
)

AVAILABLE_CASES = {
    "safepackage": "1337.0.2",
    "equests":     "2.27.1",
    "r4quests":    "2.28.1",
}
PACKAGE = sys.argv[1] if len(sys.argv) > 1 else "safepackage"
VERSION = AVAILABLE_CASES.get(PACKAGE)
if VERSION is None:
    raise SystemExit(
        f"unknown case: {PACKAGE}. Choose one of {list(AVAILABLE_CASES)}"
    )
CASE_DIR = ROOT / "data" / "example_package" / f"{PACKAGE}-{VERSION}"
ORIGIN_LOG = ROOT / "data" / "pa_logs" / f"{PACKAGE}_origin_execute.json"
ORIGINAL_SRC = CASE_DIR / "original"
OUTPUT_DIR = CASE_DIR / "patched_runtime"


def call_openai(system: str, user: str, model: str = "gpt-4.1-mini", temperature: float = 1.0) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("openai package is required. pip install openai") from e

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def load_origin_stderr(path: Path) -> str:
    doc = json.loads(path.read_text(encoding="utf-8"))
    stderr = doc["Analysis"]["execute"].get("Stderr") or ""
    try:
        return base64.b64decode(stderr).decode("utf-8", errors="replace")
    except Exception:
        return stderr


def read_package_file(rel_path: str) -> str:
    """Resolve a Stage-1 relative path against the shipped original tree."""
    target = ORIGINAL_SRC / rel_path
    if target.is_file():
        return target.read_text(encoding="utf-8")
    return ""


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("[!] Set OPENAI_API_KEY first.")
        sys.exit(1)

    stderr = load_origin_stderr(ORIGIN_LOG)

    # Error-type classification (Python built-in exception hierarchy).
    err_info = ExecuteStageErrorMessageProcessor().classify_error(stderr)
    err_type = err_info.get("error_type", "unknown")
    print(f"[0] classified error type = {err_type}")

    # Stage 1: locate primary error file (and optional caller/callee frames).
    stage1_prompt = Stage1PromptGenerator().generate_prompt(
        package_name=PACKAGE,
        package_version=VERSION,
        error_detail=stderr,
    )
    stage1_resp = call_openai(
        system="You are a Python error analysis expert.",
        user=stage1_prompt,
    )
    stage1 = Stage1PromptGenerator.parse_response(stage1_resp)
    if not stage1 or "primary_error_file" not in stage1:
        print("[!] Stage 1 could not parse LLM response:")
        print(stage1_resp)
        sys.exit(2)
    rel_path = stage1["primary_error_file"]
    line_number = int(stage1["line_number"])
    related_frames = stage1.get("related_frames") or []
    print(f"[1] primary_error_file = {rel_path}  line = {line_number}")
    if related_frames:
        frames_summary = ", ".join(
            f"{f.get('role','related')}:{f.get('file')}:{f.get('line')}"
            for f in related_frames[:2]
        )
        print(f"[1] related_frames     = {frames_summary}")

    target_path = ORIGINAL_SRC / rel_path
    source_text = target_path.read_text(encoding="utf-8")

    # Stage 2: assemble context, including caller/callee snippets when
    # Stage 1 reported them, and request a patch.
    ctx_body = CodeContextExtractor.extract_file_content(source_text, line_number)
    file_summary = CodeContextExtractor.extract_file_summary(source_text)
    related_context = CodeContextExtractor.extract_related_context(
        related_frames, read_package_file,
    )
    stage2_prompt = Stage2PromptGenerator().generate_prompt(
        error_type=err_type,
        primary_error_file=rel_path,
        line_number=line_number,
        file_content=ctx_body["body"],
        file_summary=file_summary,
        related_context=related_context,
        start_line=ctx_body["start"],
        end_line=ctx_body["end"],
    )
    stage2_resp = call_openai(
        system="You are a Python error resolution expert.",
        user=stage2_prompt,
    )
    patched_region = Postprocessor.strip_markdown_fences(stage2_resp)

    # Stitch the patched region back into the full file.
    src_lines = source_text.splitlines()
    head = src_lines[: ctx_body["start"] - 1]
    tail = src_lines[ctx_body["end"]:]
    final_code = "\n".join(head + patched_region.splitlines() + tail) + "\n"

    ok, err = Postprocessor.validate_syntax(final_code)
    if not ok:
        print(f"[!] compile() rejected the patch: {err}")
        sys.exit(3)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final_code, encoding="utf-8")
    print(f"[done] patched file written to {out_path}")
    print(
        "       diff against "
        f"data/example_package/{PACKAGE}-{VERSION}/patched/{rel_path}"
    )


if __name__ == "__main__":
    main()
