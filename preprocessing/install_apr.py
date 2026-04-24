"""
LADAR: Install-phase APR -- public release version.

This module provides the core classification and prompt-generation logic
for repairing installation-phase failures, aligned with the nine error
categories reported in the paper (Table 6).

Components
----------
- RefinedErrorMessageProcessor:
    Classifies a PA-tool stderr message into one of eight named
    categories (or leaves it as 'unknown' when no pattern matches).

- TargetedPromptGenerator:
    Builds an LLM repair prompt from a single label-agnostic template.
    Role and rules do not branch on the error category or the package
    label; the matched category is passed only through the error_type
    placeholder in the prompt.

The categories are:
    1. parsing_error
    2. missing_source_code
    3. missing_library
    4. missing_python_versions
    5. dependency_conflicts
    6. missing_configuration_files
    7. inconsistent_configurations_with_metadata
    8. no_matching_distribution
    9. unknown (reporting-only)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────
# Error classification
# ─────────────────────────────────────────────────────────────────────

class RefinedErrorMessageProcessor:
    """Classifies PA-tool stderr messages into one of the paper's
    install-phase error categories."""

    def __init__(self) -> None:
        self.normalization_patterns = [
            (r'/[^\s]+/([^/\s]+)$', r'/\1'),
            (r'[A-Z]:\\[^\s]+\\([^\\]+)$', r'\\\1'),
            (r'\b\d+\.\d+\.\d+\b', ''),
            (r'\b\d+\.\d+\b', ''),
            (r'([a-zA-Z0-9_-]+)-\d+\.\d+\.\d+', r'\1-'),
            (r'line \d+', 'line '),
            (r'\b[a-f0-9]{32,}\b', ''),
            (r'\b[A-Za-z0-9+/]{50,}={0,2}\b', ''),
            (r'\b\d{1,3}(?:\.\d{1,3}){3}\b', ''),
            (r'/tmp/[^\s]+', ''),
            (r'/var/folders/[^\s]+', ''),
        ]

        # 8 named categories; 'unknown' is returned implicitly when
        # none of the patterns match. Priorities are the original
        # ordering carried over from the experimental pipeline.
        self.ERROR_PATTERNS: Dict[str, Dict] = {
            'parsing_error': {
                'patterns': [
                    r"invalid syntax",
                    r"SyntaxError:",
                    r"unexpected EOF while parsing",
                    r"invalid character in identifier",
                    r"Error compiling Cython file",
                    r"CompileError",
                ],
                'priority': 1,
                'basis': "PyConf 10 -- Parsing error",
            },
            'missing_source_code': {
                'patterns': [
                    r"Missing source code",
                    r"Creating placeholders",
                    r"FileNotFoundError: .*No such file or directory",
                    r"No such file or directory: '.*\.py'",
                    r"No such file or directory: '.*\.rst'",
                    r"No such file or directory: '.*\.md'",
                ],
                'priority': 2,
                'basis': "PyConf 9 -- Missing source code",
            },
            # Union of the original 'missing_required_libraries_for_setup'
            # and 'module_not_found' patterns. Both indicate that a required
            # library or module could not be located.
            'missing_library': {
                'patterns': [
                    r"ModuleNotFoundError: No module named '([^']+)'",
                    r"ImportError: No module named ([^\s]+)",
                    r"ModuleNotFoundError:\s*No module named '([^']+)'",
                    r"ImportError:\s*No module named ([^\s]+)",
                    r"Could not find a version that satisfies the requirement ([^\s]+)",
                    r"ERROR: Could not build wheels for ([^\s]+)",
                    r"ctypes\.util\.find_library\(\) did not manage to locate a library called",
                ],
                'priority': 3,
                'basis': "PyConf 2/6 -- Missing required libraries or modules",
            },
            'dependency_conflicts': {
                'patterns': [
                    r"VersionConflict",
                    r"pkg_resources\.VersionConflict",
                    r"Could not resolve dependencies",
                    r"requires a different Python",
                    r"Unsolvable constraints",
                    r"conflicting dependency",
                    r"Requirement\.parse",
                ],
                'priority': 4,
                'basis': "PyConf 4 -- Dependency conflicts",
            },
            'missing_configuration_files': {
                'patterns': [
                    r"does not appear to be a Python project",
                    r"neither 'setup\.py' nor 'pyproject\.toml' found",
                    r"setup\.py: No such file or directory",
                    r"pyproject\.toml.*No such file or directory",
                    r"Project configuration file .* not found",
                ],
                'priority': 5,
                'basis': "PyConf 1 -- Incomplete configuration",
            },
            'missing_python_versions': {
                'patterns': [
                    r"requires a different Python: ([\d\.]+) not in '([^']+)'",
                    r"Package '[^']+' requires a different Python",
                    r"python_requires.*?requires-python",
                    r"SyntaxError:\s*Missing parentheses in call to ['\"]print['\"].*",
                    r"[Rr]equires Python \d+\.\d+ or better\. Python \d+\.\d+ detected",
                ],
                'priority': 6,
                'basis': "PyConf 5 -- Incorrect Python version constraint",
            },
            'inconsistent_configurations_with_metadata': {
                'patterns': [
                    r"metadata-generation-failed",
                    r"python setup\.py egg_info did not run successfully",
                    r"Preparing metadata \(setup\.py\): finished with status 'error'",
                ],
                'priority': 7,
                'basis': "PyConf 7 -- Inconsistent configurations with metadata",
            },
            'no_matching_distribution': {
                'patterns': [
                    r"No matching distribution found for ([^\s]+)",
                    r"Could not find a version that satisfies the requirement ([^\s]+)",
                ],
                'priority': 8,
                'basis': "No matching distribution on index",
            },
        }

    # ─────────────────────────────────────────────────────────────
    # Classification
    # ─────────────────────────────────────────────────────────────
    def classify_error(self, error_message: str) -> Dict:
        """Returns the highest-priority matching category.

        Falls back to 'SubprocessError' when the outer layer is
        'subprocess-exited-with-error' but the inner traceback does not
        match any pattern; otherwise returns an empty error_type (which
        is reported as 'unknown' downstream).
        """
        sorted_items = sorted(
            self.ERROR_PATTERNS.items(),
            key=lambda kv: kv[1]['priority'],
        )
        for error_type, cfg in sorted_items:
            for pat in cfg['patterns']:
                m = re.search(pat, error_message, re.IGNORECASE | re.MULTILINE)
                if m:
                    return {
                        'error_type': error_type,
                        'match': m,
                        'extracted': list(m.groups()) if m.groups() else [],
                    }

        if 'subprocess-exited-with-error' in error_message:
            tb_match = re.search(
                r'Traceback.*?(?=\[end of output\]|$)',
                error_message, re.DOTALL | re.IGNORECASE,
            )
            if tb_match:
                return self.classify_error(tb_match.group(0))

        return {'error_type': '', 'match': None, 'extracted': []}

    def extract_error_core(self, error_message: str) -> Dict:
        """Full extraction pass: classification + traceback capture +
        message normalization.
        """
        result = {
            'original': error_message,
            'normalized': None,
            'error_type': None,
            'key_info': {},
            'has_traceback': False,
            'traceback_lines': [],
        }

        classification = self.classify_error(error_message)
        etype = classification.get('error_type') or 'unknown'
        result['error_type'] = etype
        if classification.get('extracted'):
            result['key_info'][etype] = classification['extracted']

        if 'Traceback' in error_message:
            result['has_traceback'] = True
            capture = False
            for line in error_message.split('\n'):
                if 'Traceback' in line:
                    capture = True
                if capture:
                    result['traceback_lines'].append(line)

            # Supplement the classification with the final exception
            # line when classification returned 'unknown'.
            for tb_line in reversed(result['traceback_lines']):
                m = re.match(r'^(\w+Error):\s*(.+)', tb_line)
                if m:
                    if result['error_type'] == 'unknown':
                        result['error_type'] = m.group(1)
                        result['key_info']['error_detail'] = m.group(2)
                    break

        if result['error_type'] == 'unknown' and 'subprocess-exited-with-error' in error_message:
            result['error_type'] = 'SubprocessError'
            err_m = re.search(r'ERROR:\s*(.+?)(?:\n|$)', error_message)
            if err_m:
                result['key_info']['subprocess_message'] = err_m.group(1)

        normalized = error_message
        for pat, repl in self.normalization_patterns:
            normalized = re.sub(pat, repl, normalized)
        result['normalized'] = normalized

        return result


# ─────────────────────────────────────────────────────────────────────
# Prompt generation
# ─────────────────────────────────────────────────────────────────────

class TargetedPromptGenerator:
    """Generates an LLM repair prompt for install-phase failures.

    Uses a single label-agnostic template. The prompt role and rules do
    not vary across error categories or between benign and malicious
    packages; category information is conveyed only through the
    ``error_type`` placeholder filled by ``RefinedErrorMessageProcessor``.
    """

    PROMPT_TEMPLATE: str = """\
You are a Python packaging and installation expert. The package installation failed.

===== ERROR INFO =====
- Error Type: {error_type}
- Details:   {error_detail}

===== Affected File: {file_path} =====
Below is the full content of the file that caused the installation failure.

{full_file_content}

===== RULES =====
1. DO NOT remove or alter existing logic -- preserve all original functional behavior
2. Preserve original code structure and all existing payloads

===== OUTPUT =====
Return ONLY the corrected contents of `{file_path}` (no explanations, no markdown fences)."""

    # ─────────────────────────────────────────────────────────────
    # Prompt rendering
    # ─────────────────────────────────────────────────────────────
    def generate_prompt(
        self,
        error_type: str,
        *,
        file_path: str,
        full_file_content: str,
        error_detail: str = "",
    ) -> str:
        """Fills the single unified template with the supplied fields.

        ``error_type`` flows into the ERROR INFO section as text; the
        role and rules do not branch on it.
        """
        return self.PROMPT_TEMPLATE.format(
            error_type=error_type or "unknown",
            error_detail=error_detail,
            file_path=file_path,
            full_file_content=full_file_content,
        )


__all__ = [
    "RefinedErrorMessageProcessor",
    "TargetedPromptGenerator",
]
