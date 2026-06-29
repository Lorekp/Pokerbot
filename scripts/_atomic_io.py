"""Canonical atomic JSON-write helper for the eval scripts.

This module exists because of a documented class of bugs in
SCRIPTS-EVAL-AUDIT (CRIT): the eval scripts that publish verdicts to S3
were writing JSON with ``open(..., 'w')`` + ``json.dump`` with no
``fsync`` and no atomic rename. A power cut or SIGKILL mid-write
truncated the verdict file; the babysitter would then upload a
PARTIAL verdict to S3 and the watcher dashboard would render a
GREEN check on a half-written file. The same hole existed in the
five Frida capture tools in ``scripts/live/`` (FRIDA-AUDIT and the
CRASH-RESUME-AUDIT #744 lineage).

All eval scripts and dashboards that persist a JSON verdict MUST
import ``write_json_atomic`` from here. The write does:

  1. write to ``<path>.atomic.tmp``
  2. ``flush`` + ``os.fsync`` so the bytes are on disk
  3. ``os.replace`` -> ``<path>`` (atomic on POSIX and on Win32
     since Python 3.3)
  4. fsync the parent directory so the rename itself is durable

The pattern mirrors ``pokerbot/solver/label_completion.py:514-546``
(the in-tree exemplar) and ``pokerbot/training/leduc_tabular_cfr.py``
(CRASH-RESUME-AUDIT #1).

Companion helpers:

  * ``write_jsonl_line`` — append a single JSON object to a JSONL
    file with ``flush`` + ``fsync`` (the live capture path).
  * ``read_json`` — read a JSON file with ``errors="replace"`` so a
    torn tail does not abort downstream consumers.

Stdlib only. No third-party dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | Path, obj: Any, *, indent: int | None = 2,
                      ensure_parents: bool = True) -> Path:
    """Atomically write ``obj`` as JSON to ``path``.

    Returns the resolved Path. Steps:

      1. mkdir parents (if ``ensure_parents``)
      2. write to ``<path>.atomic.tmp``
      3. flush + fsync the tmp fd
      4. ``os.replace`` -> ``<path>`` (atomic rename)
      5. fsync the parent directory so the rename is durable

    Safe against SIGKILL/power-cut mid-write: a partial tmp file is
    never visible at ``<path>`` and is overwritten on the next call.

    No-op on missing parent (raises FileNotFoundError).
    """
    p = Path(path).resolve()
    if ensure_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".atomic.tmp")
    payload = json.dumps(obj, indent=indent, ensure_ascii=False)
    # newline-terminated so ``cat | jq`` works on every platform
    if not payload.endswith("\n"):
        payload += "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, p)
    # Parent-dir fsync so the rename itself is durable on POSIX.
    # On Windows, os.replace IS atomic AND GetFileInformationByHandle
    # flushes the metadata, so the dir fsync is best-effort only.
    try:
        dir_fd = os.open(str(p.parent), os.O_RDONLY)
    except OSError:
        dir_fd = -1
    if dir_fd >= 0:
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)
    return p


def write_jsonl_line(path: str | Path, obj: Any, *, fsync: bool = True) -> None:
    """Append one JSON object as a single JSONL line.

    Flushes after every line (matches the live capture path's existing
    behavior). When ``fsync`` is True (default), also ``fsync`` the fd
    so a power-cut after ``write_jsonl_line`` returns does not lose
    the line. This is the LIVE-CAPTURE durability knob — the CRASH-
    RESUME-AUDIT wired ``fsync`` into HandRecorder + DecisionLogger
    but NOT into the standalone Frida capture tools; this helper lets
    those tools pick up the same guarantee.

    Appends ``\n`` so the file is one-line-per-record JSONL.
    """
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.flush()
        if fsync:
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Read a JSON file with ``errors='replace'`` and a torn-tail safety net.

    A truncated JSON file (the bug class this whole module was created
    to prevent) used to raise ``json.JSONDecodeError`` on every reader.
    This helper:

      * opens with ``errors="replace"`` so a partial UTF-8 sequence
        does not raise ``UnicodeDecodeError``
      * returns ``default`` when the file is missing, empty, or
        unparseable

    For strict JSONL readers prefer iterating the file directly with
    ``errors="replace"``; this helper is for one-shot JSON verdicts.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return default
    text = text.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return default
