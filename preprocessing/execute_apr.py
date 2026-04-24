"""
LADAR: Execute-phase APR -- public release version.

This module provides the core classification, prompt-generation, and
post-processing logic for repairing execution-phase failures. It
implements the two-stage repair process described in the paper:

  Stage 1 -- Error analysis: identify the primary file and line to fix
             from the traceback, and (optionally) up to two caller or
             callee frames for multi-file call chains.

  Stage 2 -- Context-based repair: rewrite the affected code region
             using the enclosing function (AST-based) or a line-based
             window when no function encloses the error line, together
             with a signature-level summary of the rest of the file and
             optional caller/callee snippets.

Components
----------
- ExecuteStageErrorMessageProcessor:
    Classifies an execute-phase stderr message using Python's built-in
    exception hierarchy (e.g., ModuleNotFoundError, TypeError,
    SyntaxError), plus a GeneralExecutionError fallback.

- Stage1PromptGenerator:
    Builds the Stage 1 error-analysis prompt and parses the JSON
    response, including the optional ``related_frames`` field.

- CodeContextExtractor:
    Produces the three pieces of code context used in Stage 2:
    ``file_content`` (the enclosing function or a +/-30-line window),
    ``file_summary`` (a signature-level summary of the rest of the
    file), and ``related_context`` (caller/callee snippets).

- Stage2PromptGenerator:
    Builds the Stage 2 context-based repair prompt with a single
    label-agnostic template. Supports the iterative APR history block.

- Postprocessor:
    Strips Markdown fences and validates patches with Python's
    ``compile()`` syntax check.

The module does not call the LLM or perform file I/O; these concerns
are left to a runner script that composes these components.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

# Long tracebacks are truncated from the head so the tail (which carries
# the actual exception type and the most informative frame) is
# preserved when sent to the Stage 1 prompt.
TRACEBACK_TAIL_CHARS: int = 4000

# Number of context lines on each side of the error line, used when no
# enclosing function can be extracted.
WINDOW_CONTEXT_LINES: int = 30

# Upper bound on characters for the signature-level file summary.
FILE_SUMMARY_MAX_CHARS: int = 2000

# Upper bound on characters for the caller/callee reference block.
RELATED_CONTEXT_MAX_CHARS: int = 2000


# ─────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────

class ExecuteStageErrorMessageProcessor:
    """Classifies an execute-phase stderr message using Python's
    built-in exception hierarchy.

    Patterns are ordered by priority (lower is higher). When no
    pattern matches, the classifier returns ``GeneralExecutionError``
    if the message contains a generic failure indicator (e.g.,
    ``Traceback``, ``Error:``), otherwise ``unknown``.
    """

    GENERAL_INDICATORS: Tuple[str, ...] = (
        "Traceback", "Error:",
    )

    def __init__(self) -> None:
        self.ERROR_PATTERNS: Dict[str, Dict] = {
            # --- Syntax / indentation ---
            "SyntaxError": {
                "patterns": [
                    r"^SyntaxError:", r"invalid syntax",
                    r"unexpected EOF while parsing",
                    r"unterminated string literal",
                    r"cannot assign to",
                ],
                "priority": 1,
            },
            "IndentationError": {
                "patterns": [
                    r"^IndentationError:", r"expected an indented block",
                    r"unindent does not match any outer indentation level",
                    r"unexpected indent",
                ],
                "priority": 2,
            },
            "TabError": {
                "patterns": [
                    r"^TabError:",
                    r"inconsistent use of tabs and spaces in indentation",
                ],
                "priority": 3,
            },
            # --- Import / module resolution ---
            "ModuleNotFoundError": {
                "patterns": [
                    r"^ModuleNotFoundError:",
                    r"No module named '([^']+)'",
                ],
                "priority": 4,
            },
            "ImportError": {
                "patterns": [
                    r"^ImportError:",
                    r"cannot import name '([^']+)'",
                ],
                "priority": 5,
            },
            # --- Attribute / name ---
            "AttributeError": {
                "patterns": [
                    r"^AttributeError:",
                    r"'([^']+)' object has no attribute '([^']+)'",
                    r"module '([^']+)' has no attribute '([^']+)'",
                ],
                "priority": 6,
            },
            "NameError": {
                "patterns": [
                    r"^NameError:",
                    r"name '([^']+)' is not defined",
                ],
                "priority": 7,
            },
            "UnboundLocalError": {
                "patterns": [
                    r"^UnboundLocalError:",
                    r"local variable '([^']+)' referenced before assignment",
                ],
                "priority": 8,
            },
            # --- Type / value ---
            "TypeError": {
                "patterns": [
                    r"^TypeError:",
                    r"'([^']+)' object is not (callable|iterable|subscriptable)",
                    r"unsupported operand type\(s\) for",
                    r"takes \d+ positional arguments? but \d+ were? given",
                    r"missing \d+ required positional arguments?",
                ],
                "priority": 9,
            },
            "ValueError": {
                "patterns": [
                    r"^ValueError:",
                    r"invalid literal for int\(\)",
                    r"could not convert string to float",
                    r"not enough values to unpack",
                ],
                "priority": 10,
            },
            # --- Lookup ---
            "IndexError": {
                "patterns": [
                    r"^IndexError:",
                    r"list index out of range",
                    r"tuple index out of range",
                    r"string index out of range",
                ],
                "priority": 11,
            },
            "KeyError": {
                "patterns": [
                    r"^KeyError:",
                ],
                "priority": 12,
            },
            # --- File system / OS ---
            "FileNotFoundError": {
                "patterns": [
                    r"^FileNotFoundError:",
                    r"\[Errno 2\] No such file or directory",
                ],
                "priority": 13,
            },
            "IsADirectoryError": {
                "patterns": [
                    r"^IsADirectoryError:",
                    r"\[Errno 21\]",
                ],
                "priority": 14,
            },
            "NotADirectoryError": {
                "patterns": [
                    r"^NotADirectoryError:",
                    r"\[Errno 20\]",
                ],
                "priority": 15,
            },
            "FileExistsError": {
                "patterns": [
                    r"^FileExistsError:",
                    r"\[Errno 17\]",
                ],
                "priority": 16,
            },
            "PermissionError": {
                "patterns": [
                    r"^PermissionError:",
                    r"\[Errno 13\]",
                ],
                "priority": 17,
            },
            # --- Network (often caused by sandbox restrictions) ---
            "ConnectionError": {
                "patterns": [
                    r"^ConnectionError:",
                    r"ConnectionRefusedError",
                    r"ConnectionResetError",
                    r"Failed to establish a new connection",
                ],
                "priority": 18,
            },
            "TimeoutError": {
                "patterns": [
                    r"^TimeoutError:",
                    r"\[Errno 110\]",
                    r"Read timed out",
                ],
                "priority": 19,
            },
            "HTTPError": {
                "patterns": [
                    r"^HTTPError:",
                    r"urllib\.error\.HTTPError",
                    r"requests\.exceptions\.HTTPError",
                ],
                "priority": 20,
            },
            # --- Runtime / assertion / misc ---
            "AssertionError": {
                "patterns": [r"^AssertionError:"],
                "priority": 21,
            },
            "RecursionError": {
                "patterns": [
                    r"^RecursionError:",
                    r"maximum recursion depth exceeded",
                ],
                "priority": 22,
            },
            "NotImplementedError": {
                "patterns": [r"^NotImplementedError:"],
                "priority": 23,
            },
            "RuntimeError": {
                "patterns": [r"^RuntimeError:"],
                "priority": 24,
            },
            "EOFError": {
                "patterns": [r"^EOFError:"],
                "priority": 25,
            },
            "SystemExit": {
                "patterns": [r"^SystemExit:"],
                "priority": 26,
            },
            "SystemError": {
                "patterns": [r"^SystemError:"],
                "priority": 27,
            },
        }

    def classify_error(self, error_message: str) -> Dict:
        """Returns the highest-priority matching category.

        If no specific pattern matches, returns ``GeneralExecutionError``
        when the message contains a generic failure indicator, and
        ``unknown`` otherwise.
        """
        sorted_items = sorted(
            self.ERROR_PATTERNS.items(),
            key=lambda kv: kv[1]["priority"],
        )
        for error_type, cfg in sorted_items:
            for pat in cfg["patterns"]:
                m = re.search(pat, error_message, re.MULTILINE)
                if m:
                    return {
                        "error_type": error_type,
                        "match": m,
                        "extracted": list(m.groups()) if m.groups() else [],
                    }

        if any(ind in error_message for ind in self.GENERAL_INDICATORS):
            return {"error_type": "GeneralExecutionError", "match": None, "extracted": []}

        return {"error_type": "unknown", "match": None, "extracted": []}


# ─────────────────────────────────────────────────────────────────────
# Stage 1: Error Analysis prompt
# ─────────────────────────────────────────────────────────────────────

class Stage1PromptGenerator:
    """Builds the Stage 1 error-analysis prompt and parses the JSON
    response.

    The prompt asks the LLM to identify the primary error file, line
    number, and error type, and, when the traceback reveals a
    multi-file call chain, up to two additional caller or callee
    frames. To keep the most informative part of long tracebacks, the
    input is tail-truncated to ``TRACEBACK_TAIL_CHARS`` characters.
    """

    PROMPT_TEMPLATE: str = """\
You are a Python error analysis expert. Analyze this error traceback to identify the PRIMARY file that should be fixed.

=========== PACKAGE INFO ===========
Package: {package_name}
Version: {package_version}

======= FULL ERROR TRACEBACK =======
{error_detail}

======= ANALYSIS RULES =======
1. IGNORE system files like:
   - /usr/local/bin/analyze-python.py
   - /usr/lib/python3.10/...
   - <frozen importlib._bootstrap>

2. FOCUS on package files:
   - Files in /usr/local/../dist-packages/{package_name}/

3. If the error involves a CALL CHAIN (caller -> callee), report up to 2 related
   package-internal files from the traceback as callers or callees of the primary error.

======= OUTPUT FORMAT =======
Return ONLY a JSON object with this exact structure:
{{
    "primary_error_file": "relative/path/to/file.py",
    "line_number": 123,
    "error_type": "ModuleNotFoundError|SyntaxError|etc",
    "reasoning": "Brief explanation why this is the primary error location",
    "related_frames": [
        {{"file": "relative/path/to/caller.py", "line": 45, "role": "caller"}}
    ]
}}
The "related_frames" array is optional. Include only if the traceback shows a multi-file call chain within the package (max 2 entries).
"""

    def generate_prompt(
        self,
        package_name: str,
        package_version: str,
        error_detail: str,
    ) -> str:
        """Fills the template. Long tracebacks are tail-truncated to
        preserve the most informative final frames.
        """
        if len(error_detail) > TRACEBACK_TAIL_CHARS:
            error_detail = error_detail[-TRACEBACK_TAIL_CHARS:]
        return self.PROMPT_TEMPLATE.format(
            package_name=package_name,
            package_version=package_version,
            error_detail=error_detail,
        )

    @staticmethod
    def parse_response(response: str) -> Optional[Dict]:
        """Parses the LLM's JSON response into a normalized dict with
        keys: ``primary_error_file``, ``line_number``, ``error_type``,
        ``reasoning``, ``related_frames``. Returns ``None`` on parse
        failure.
        """
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            data = json.loads(match.group() if match else response)

            related_raw = data.get("related_frames") or []
            related: List[Dict] = []
            for frame in list(related_raw)[:2]:
                if not isinstance(frame, dict):
                    continue
                if "file" not in frame or "line" not in frame:
                    continue
                try:
                    related.append({
                        "file": str(frame["file"]),
                        "line": int(frame["line"]),
                        "role": str(frame.get("role", "related")),
                    })
                except (ValueError, TypeError):
                    continue

            return {
                "primary_error_file": str(data.get("primary_error_file", "")),
                "line_number": int(data.get("line_number", 1)),
                "error_type": str(data.get("error_type", "")),
                "reasoning": str(data.get("reasoning", "")),
                "related_frames": related,
            }
        except (AttributeError, json.JSONDecodeError, ValueError, KeyError, TypeError):
            return None


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Code context extraction
# ─────────────────────────────────────────────────────────────────────

class CodeContextExtractor:
    """Extracts the three pieces of code context used in Stage 2.

    ``file_content``: the enclosing function of the error line, or a
    +/-``WINDOW_CONTEXT_LINES`` line window when no function encloses
    the line (e.g., top-level code).

    ``file_summary``: signature-level summary of the rest of the file,
    always attached so the model can reason about the broader file
    organization at low token cost.

    ``related_context``: snippets from caller or callee frames
    identified by Stage 1 for multi-file call chains, supplied as
    reference context only.
    """

    # --- file_content ---------------------------------------------------

    @staticmethod
    def extract_file_content(source_code: str, line_number: int) -> Dict:
        """Returns a dict with ``start``, ``end``, ``body``, ``kind``,
        and (for functions) ``name``. ``kind`` is either ``function``
        or ``window``.
        """
        func_info = CodeContextExtractor._extract_enclosing_function(
            source_code, line_number,
        )
        if func_info is not None:
            return func_info

        lines = source_code.splitlines()
        if not lines:
            return {"start": 1, "end": 1, "body": "", "kind": "window"}

        target = max(1, min(line_number, len(lines)))
        start = max(1, target - WINDOW_CONTEXT_LINES)
        end = min(len(lines), target + WINDOW_CONTEXT_LINES)
        body = "\n".join(lines[start - 1: end])
        return {"start": start, "end": end, "body": body, "kind": "window"}

    @staticmethod
    def _extract_enclosing_function(
        source_code: str,
        line_number: int,
    ) -> Optional[Dict]:
        """AST-based: returns the smallest function/method whose span
        encloses ``line_number``.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return None

        best: Optional[Dict] = None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = node.lineno
            end = node.end_lineno or start
            if start <= line_number <= end:
                if best is None or (end - start) < (best["end"] - best["start"]):
                    best = {"name": node.name, "start": start, "end": end}

        if best is None:
            return None

        lines = source_code.splitlines()
        body = "\n".join(lines[best["start"] - 1: best["end"]])
        return {
            "name": best["name"],
            "start": best["start"],
            "end": best["end"],
            "body": body,
            "kind": "function",
        }

    # --- file_summary ---------------------------------------------------

    @staticmethod
    def extract_file_summary(
        source_code: str,
        max_chars: int = FILE_SUMMARY_MAX_CHARS,
    ) -> str:
        """Returns a signature-level summary of top-level functions and
        classes. Always produced when possible; empty string when the
        file cannot be parsed.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return ""

        funcs: List[str] = []
        classes: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                funcs.append(f"#   def {node.name}({', '.join(args)})")
            elif isinstance(node, ast.ClassDef):
                bases = [getattr(b, "id", "?") for b in node.bases]
                base_str = f"({', '.join(bases)})" if bases else ""
                classes.append(f"#   class {node.name}{base_str}")
                methods = [
                    c for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                for m in methods[:5]:
                    classes.append(f"#     - def {m.name}()")

        parts: List[str] = []
        if funcs:
            parts.append("EXISTING FUNCTIONS:")
            parts.extend(funcs[:15])
        if classes:
            parts.append("EXISTING CLASSES:")
            parts.extend(classes[:15])

        summary = "\n".join(parts)
        return summary[:max_chars] if summary else ""

    # --- related_context ------------------------------------------------

    @staticmethod
    def extract_related_context(
        related_frames: List[Dict],
        read_file_fn: Callable[[str], str],
        max_chars: int = RELATED_CONTEXT_MAX_CHARS,
    ) -> str:
        """Returns a concatenated reference block assembled from the
        caller or callee frames reported by Stage 1.

        ``read_file_fn`` is supplied by the runner and must return the
        full text of a package-internal file given its relative path
        (as reported by Stage 1). An empty string signals that the
        file cannot be read.
        """
        if not related_frames:
            return ""

        parts: List[str] = []
        total = 0
        for frame in related_frames[:2]:
            try:
                content = read_file_fn(frame["file"])
            except Exception:
                content = ""
            if not content:
                continue

            func_info = CodeContextExtractor._extract_enclosing_function(
                content, frame["line"],
            )
            if func_info is not None:
                snippet = func_info["body"]
                label = (
                    f"{frame.get('role', 'related')} function "
                    f"'{func_info.get('name', '')}' in {frame['file']}"
                )
            else:
                lines = content.splitlines()
                s = max(0, frame["line"] - 11)
                e = min(len(lines), frame["line"] + 10)
                snippet = "\n".join(lines[s:e])
                label = (
                    f"{frame.get('role', 'related')} near line "
                    f"{frame['line']} in {frame['file']}"
                )

            remaining = max_chars - total
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                snippet = snippet[:remaining]

            parts.append(f"[{label}]\n{snippet}")
            total += len(snippet)

        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Stage 2: Repair prompt
# ─────────────────────────────────────────────────────────────────────

class Stage2PromptGenerator:
    """Builds the Stage 2 context-based code repair prompt.

    Uses a single label-agnostic template. The prompt role and rules
    do not vary between benign and malicious packages; the matched
    error category is passed only through the ``error_type`` field.

    Supports the iterative APR history block: when prior repair
    attempts are supplied, a ``PREVIOUS ATTEMPTS`` section is appended
    together with a directive asking the LLM to try a different
    strategy.
    """

    PROMPT_TEMPLATE: str = """\
You are a Python error resolution expert. Fix the following Python error.

======= ERROR INFO =======
- Error Type: {error_type}
- File: {primary_error_file}
- Line: {line_number}

======= Affected File Content =======
# FILE STRUCTURE SUMMARY
{file_summary}
# END STRUCTURE SUMMARY
{file_content}

======= Related Code (reference only) =======
{related_context}
{prev_block}
======= RULES =======
1. DO NOT remove or alter existing logic -- only fix the error
2. Preserve the original code's intended behavior and all existing functionality

======= OUTPUT =======
Return ONLY the corrected lines {start_line} to {end_line} (no explanations, no markdown fences)."""

    PREV_BLOCK_HEADER: str = "\n===== PREVIOUS ATTEMPTS ====="
    PREV_BLOCK_FOOTER: str = "\nThe above attempts failed. Try a DIFFERENT approach."

    # Per-round character caps mirroring the paper's Listing 10 block.
    PREV_PATCH_MAX_CHARS: int = 2000
    PREV_ERROR_MAX_CHARS: int = 1000

    def generate_prompt(
        self,
        *,
        error_type: str,
        primary_error_file: str,
        line_number: int,
        file_content: str,
        file_summary: str,
        related_context: str,
        start_line: int,
        end_line: int,
        prev_attempts: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Renders the prompt. ``prev_attempts`` is a list of
        ``(patch_text, pa_error)`` pairs from earlier rounds, in
        chronological order.
        """
        prev_block = self._format_prev_block(prev_attempts)
        return self.PROMPT_TEMPLATE.format(
            error_type=error_type or "unknown",
            primary_error_file=primary_error_file,
            line_number=line_number,
            file_summary=file_summary,
            file_content=file_content,
            related_context=related_context or "(none)",
            prev_block=prev_block,
            start_line=start_line,
            end_line=end_line,
        )

    def _format_prev_block(
        self,
        prev_attempts: Optional[List[Tuple[str, str]]],
    ) -> str:
        if not prev_attempts:
            return ""
        lines: List[str] = [self.PREV_BLOCK_HEADER, ""]
        for i, (patch, pa_error) in enumerate(prev_attempts, 1):
            lines.append(f"--- Attempt {i} ---")
            lines.append("Patch applied:")
            lines.append(patch[: self.PREV_PATCH_MAX_CHARS])
            lines.append("Result error:")
            lines.append(pa_error[: self.PREV_ERROR_MAX_CHARS])
            lines.append("")
        lines.append(self.PREV_BLOCK_FOOTER.lstrip("\n"))
        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────

class Postprocessor:
    """Strips Markdown fences and validates patches with Python's
    ``compile()`` syntax check. Placeholder restoration is assumed to
    be performed by the caller using the same utility it used in
    preprocessing.
    """

    @staticmethod
    def strip_markdown_fences(text: str) -> str:
        """Removes an outer pair of triple-backtick fences if present.
        Handles both bare ``` and ```python fences.
        """
        s = text.strip()
        if s.startswith("```python"):
            s = s[len("```python"):].lstrip("\n")
            if s.endswith("```"):
                s = s[:-3].rstrip()
            return s
        if s.startswith("```") and s.endswith("```"):
            parts = s.split("\n")
            if len(parts) >= 2:
                return "\n".join(parts[1:-1])
        return s

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Returns ``(ok, error_message)``. ``ok`` is ``True`` when
        ``compile()`` succeeds. ``error_message`` is a short
        description suitable for feeding back to the LLM on retry.
        """
        try:
            compile(code, "<patch>", "exec")
            return True, None
        except SyntaxError as e:
            if e.lineno is not None and e.msg:
                return False, f"line {e.lineno}: {e.msg}"
            return False, str(e)


__all__ = [
    "ExecuteStageErrorMessageProcessor",
    "Stage1PromptGenerator",
    "CodeContextExtractor",
    "Stage2PromptGenerator",
    "Postprocessor",
    "TRACEBACK_TAIL_CHARS",
    "WINDOW_CONTEXT_LINES",
    "FILE_SUMMARY_MAX_CHARS",
    "RELATED_CONTEXT_MAX_CHARS",
]
