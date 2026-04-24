"""Serialize a PA execute JSON into a token sequence.

The same serialization is used by both the TF-IDF and Nebula detectors,
following the PA channel -> Nebula event mapping in Table 2 of the paper:

    Analysis.execute.Files    -> file_access  <path> <R|W|D>
    Analysis.execute.Sockets  -> network_connect <addr>:<port>
    Analysis.execute.DNS      -> dns_query <hostname> <type>
    Analysis.execute.Commands -> apis <execute_cmd> <arg1> <arg2> ...

File access modes are concatenated with space separation as R/W/D flags,
so that e.g. a read+write on ~/.ssh/id_rsa is represented as
``file_access /root/.ssh/id_rsa R W``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Union


def _iter_dns_events(dns_arr) -> Iterable[str]:
    if not dns_arr:
        return
    for entry in dns_arr:
        queries = (entry or {}).get("Queries") or []
        for q in queries:
            host = (q or {}).get("Hostname") or ""
            types = (q or {}).get("Types") or []
            t = types[0] if types else "A"
            if host:
                yield f"dns_query {host} {t}"


def _iter_socket_events(sockets_arr) -> Iterable[str]:
    if not sockets_arr:
        return
    for s in sockets_arr:
        addr = (s or {}).get("Address") or ""
        port = (s or {}).get("Port") or ""
        if addr:
            yield f"network_connect {addr}:{port}"


def _iter_command_events(cmd_arr) -> Iterable[str]:
    if not cmd_arr:
        return
    for entry in cmd_arr:
        parts = (entry or {}).get("Command") or []
        parts = [str(x) for x in parts]
        if parts:
            yield "apis " + " ".join(parts)


def _iter_file_events(files_arr) -> Iterable[str]:
    if not files_arr:
        return
    for f in files_arr:
        path = (f or {}).get("Path") or ""
        flags = []
        if f.get("Read"):
            flags.append("R")
        if f.get("Write"):
            flags.append("W")
        if f.get("Delete"):
            flags.append("D")
        if path:
            yield ("file_access " + path + (" " + " ".join(flags) if flags else "")).rstrip()


def pa_execute_to_tokens(pa_json: Union[dict, str, Path]) -> List[str]:
    """Return an ordered list of token lines for one PA JSON document.

    Accepts either a parsed dict or a path to a JSON file.
    """
    if isinstance(pa_json, (str, Path)):
        pa_json = json.loads(Path(pa_json).read_text(encoding="utf-8"))
    exe = (pa_json.get("Analysis") or {}).get("execute") or {}
    tokens: List[str] = []
    tokens.extend(_iter_file_events(exe.get("Files")))
    tokens.extend(_iter_socket_events(exe.get("Sockets")))
    tokens.extend(_iter_dns_events(exe.get("DNS")))
    tokens.extend(_iter_command_events(exe.get("Commands")))
    return tokens


def pa_execute_to_document(pa_json: Union[dict, str, Path]) -> str:
    """Return the single-document string TF-IDF ingests (tokens joined with \\n)."""
    return "\n".join(pa_execute_to_tokens(pa_json))
