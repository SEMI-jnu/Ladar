"""Execution-log inputs used by the reported classical and Nebula models.

Pure classical serialization functions are copied from the experiment pipeline.
Nebula uses its separate PA normalization through pa_execute_to_nebula_document.
"""
from __future__ import annotations
import json
import re
import tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

MAX_DNS_TOKENS = 400
MAX_CMD_TOKENS = 800
MAX_SOCK_TOKENS = 400
MAX_FILE_TOKENS = 1000
MAX_ROW_CHARS = 300_000
STOP = {"build", "metadata", "cfg", "utils", "util", "dist", "info", "setup", "log", "errors", "__pycache__", "local", "tmp", "http"}
PAT_PATH = re.compile(r"[\\/]")

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, tuple): return list(x)
    if isinstance(x, dict): return [x]
    try:
        if isinstance(x, (np.ndarray, pd.Series)): return list(x)
    except: pass
    return [x]

def _to_str(x, limit=None):
    if x is None: return ""
    s = x.decode("utf-8", "ignore") if isinstance(x, bytes) else str(x)
    return s[:limit] if limit and len(s) > limit else s

def any_get(d, keys, default=None):
    if not isinstance(d, dict): return default
    kl = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = kl.get(str(k).lower())
        if kk is not None: return d[kk]
    return default

def _dns_to_text(lst, cap=MAX_DNS_TOKENS):
    items = _as_list(lst)
    toks, n = [], 0
    for it in items:
        if isinstance(it, dict):
            for _, v in it.items():
                if isinstance(v, (list, tuple)):
                    for e in _as_list(v):
                        if n < cap: toks.append(f"dns:{_to_str(e)}"); n += 1
                else:
                    if n < cap: toks.append(f"dns:{_to_str(v)}"); n += 1
        else:
            if n < cap: toks.append(f"dns:{_to_str(it)}"); n += 1
        if n >= cap: break
    return " ".join(toks)

def _commands_to_text(lst, cap=MAX_CMD_TOKENS):
    items = _as_list(lst)
    out, n = [], 0
    for it in items:
        s = None
        if isinstance(it, dict):
            cmd = any_get(it, ["Command", "command", "cmd", "tokens"])
            if isinstance(cmd, (list, tuple)):
                s = " ".join(str(x) for x in _as_list(cmd))
            elif isinstance(cmd, str):
                s = cmd
        else:
            s = str(it)
        if s: out.append(_to_str(s, 256)); n += 1
        if n >= cap: break
    return " ".join(out)

def _sockets_to_text(lst, cap=MAX_SOCK_TOKENS):
    items = _as_list(lst)
    out, n = [], 0
    for it in items:
        if isinstance(it, dict):
            hs = any_get(it, ["Hostnames", "Hostname", "Hosts", "Domains"], [])
            for h in _as_list(hs):
                if n < cap: out.append(_to_str(h)); n += 1
        else:
            if n < cap: out.append(_to_str(it)); n += 1
        if n >= cap: break
    return " ".join(out)

def _files_to_text(lst, cap=MAX_FILE_TOKENS):
    items = _as_list(lst)
    out, n = [], 0
    for it in items:
        if isinstance(it, dict):
            path = any_get(it, ["Path", "path", "FilePath"])
            if path: out.append(_to_str(path))
            if any_get(it, ["Read", "read"], False): out.append("op_read")
            if any_get(it, ["Write", "write"], False): out.append("op_write")
            if any_get(it, ["Delete", "delete"], False): out.append("op_delete")
        else:
            out.append(_to_str(it))
        n += 1
        if n >= cap: break
    return " ".join(out)

def phase_to_text(phase_dict: dict) -> str:
    if not isinstance(phase_dict, dict): return ""
    pieces, used = [], 0
    def _append(txt):
        nonlocal used
        if not txt: return
        remain = MAX_ROW_CHARS - used
        if remain <= 0: return
        pieces.append(txt[:remain]); used += len(txt[:remain])
    for seg in [_dns_to_text(any_get(phase_dict, ["DNS", "dns"], [])),
                _commands_to_text(any_get(phase_dict, ["Commands", "commands"], [])),
                _sockets_to_text(any_get(phase_dict, ["Sockets", "sockets"], [])),
                _files_to_text(any_get(phase_dict, ["Files", "files"], []))]:
        if seg: _append(seg + " ")
        if used >= MAX_ROW_CHARS: break
    return "".join(pieces).rstrip()

def keep_signal_tokens(text):
    toks = (text or "").split()
    out = []
    for t in toks:
        if not t: continue
        if PAT_PATH.search(t):
            b = t.split("/")[-1].split("\\")[-1]
            if len(b) >= 3: out.append(b); continue
        if t.startswith(("dns:", "ip:")): out.append(t); continue
        if t.startswith("--") or (t.startswith("-") and len(t) > 1): out.append(t); continue
        if "." in t and re.search(r"[A-Za-z]", t): out.append(t); continue
        if re.search(r"[A-Za-z]", t) and len(t) >= 3 and t.lower() not in STOP: out.append(t)
    return out

def _plain(value):
    # ORIGIN records imported from BigQuery may contain NumPy arrays.
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _read(pa_json):
    if isinstance(pa_json, (str, Path)):
        pa_json = json.loads(Path(pa_json).read_text(encoding="utf-8"))
    return _plain(pa_json)


def pa_execute_to_document(pa_json) -> str:
    """Classical TF-IDF input; retain an empty string when execution is absent."""
    data = _read(pa_json)
    return phase_to_text((data.get("Analysis") or {}).get("execute") or {})


def pa_execute_to_tokens(pa_json) -> list[str]:
    """Classical signal tokens, before TF-IDF vocabulary filtering."""
    return keep_signal_tokens(pa_execute_to_document(pa_json))


@lru_cache(maxsize=1)
def _nebula_preprocessor():
    from .pa_nebula import PAToNebulaPreprocessor
    return PAToNebulaPreprocessor(output_dir=tempfile.gettempdir())


def pa_execute_to_nebula_document(pa_json) -> str:
    """Package metadata and execution events, as in Table 2 (no other phases)."""
    data = _read(pa_json)
    view = {"Package": data.get("Package") or {},
            "Analysis": {"execute": (data.get("Analysis") or {}).get("execute") or {}}}
    return _nebula_preprocessor().convert_pa_to_nebula_format(view)
