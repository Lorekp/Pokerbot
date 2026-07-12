"""Production loader for the SortedSnapshotDir mmap blueprint index (tier-1 serve flip, 2026-07-12).

The tier-1 126k blueprint (snap_avg(iter_126000 preflop + NLSNAP01 snapshots 100k..125k postflop),
538,848,758 rows) cannot ship as a pickle dict: the 25M-row pkl cap was the measured self-play miss
driver, and the raw snapshot-avg ckpt must NEVER be filtered/served. The serve object is the
``scripts/nlcfr_key_index.py --with-rows`` output directory — lexicographically sorted S16 keys +
aligned f32 (n,11) rows, both memory-mapped (~0 RSS); ``get`` is a binary search. This class is the
production twin of ``scripts/engine_ab_h2h.SortedSnapshotDir`` (kept out of scripts/ so the deployed
serve never imports eval-harness modules with env side effects).

CONFIG-FINGERPRINT GUARD (fail-closed): a ``serve_config.json`` stamped into the index dir records
the training/keying config the object was built and CERTIFIED under (cert7 raw +344.6/deck,
+664.9/deck with K=2000 search vs incumbent; cert8 exploitability null). ``load_blueprint_index_dir``
refuses to serve unless (a) every ``env_require`` var matches the live process env EXACTLY and
(b) the stamped ``abstraction`` dict matches the import-latched encoder config
(``pokerbot.eval.lbr.check_blueprint_serve_env``, strict). A skew on any of these keys is a silent
100%-miss (uniform-fallback play) or wrong-row HIT — the historical -559 bb/100 failure class —
so we raise instead of warn.
"""
from __future__ import annotations

import json
import os

__all__ = ["SortedSnapshotDir", "load_blueprint_index_dir"]

_REQUIRED_FILES = ("keys_sorted.bin", "rows_sorted.f32.bin", "meta.json", "serve_config.json")


class SortedSnapshotDir:
    """FULL-table mmap blueprint serve object. Duck-types the only access every serve consumer
    uses (``snapshot.get(key)`` -> f32[11] row or default; see lbr.make_blueprint_policy)."""

    def __init__(self, dirpath: str):
        import numpy as _np

        self._np = _np
        self.dirpath = dirpath
        self.keys = _np.memmap(os.path.join(dirpath, "keys_sorted.bin"), dtype="S16", mode="r")
        rows_path = os.path.join(dirpath, "rows_sorted.f32.bin")
        if not os.path.exists(rows_path):
            raise FileNotFoundError(
                f"{dirpath}: rows_sorted.f32.bin missing — build the index with "
                "nlcfr_key_index.py --with-rows (a keys-only index cannot serve values)")
        self.rows = _np.memmap(rows_path, dtype=_np.float32, mode="r").reshape(-1, 11)
        if self.rows.shape[0] != self.keys.shape[0]:
            raise ValueError(
                f"{dirpath}: keys/rows row-count mismatch "
                f"({self.keys.shape[0]:,} vs {self.rows.shape[0]:,}) — corrupt/partial download")

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    # numpy 'S16' semantics: trailing NULs ignored; keep comparisons numpy-vs-numpy (both sides
    # NUL-stripped) — see scripts/serve_miss_probe.SortedSnapshot for the false-MISS footgun.
    def get(self, key, default=None):
        kk = self._np.asarray(key, dtype="S16")
        i = int(self._np.searchsorted(self.keys, kk))
        if i < self.keys.shape[0] and self.keys[i] == kk:
            return self.rows[i]
        return default

    def __contains__(self, key) -> bool:
        return self.get(key) is not None


def load_blueprint_index_dir(path: str) -> tuple["SortedSnapshotDir", dict]:
    """Load + FAIL-CLOSED-verify a SortedSnapshotDir blueprint index for the deployed serve.

    Returns ``(snapshot, info)`` where ``info`` carries the stamped serve_config + meta for the
    caller's provenance banner. Raises RuntimeError/FileNotFoundError on ANY config skew or
    incomplete download — never serve a half-verified full-table object."""
    missing = [f for f in _REQUIRED_FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"blueprint index dir {path} incomplete: missing {missing} — re-download the index "
            "(serve_config.json is stamped at deploy; see artifacts/serve_flip_tier1.md)")

    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(path, "serve_config.json"), "r", encoding="utf-8") as f:
        stamp = json.load(f)

    # 1) EXACT env requirements (tree/keying gates that live OUTSIDE the encoder stamp: MRR
    #    action-thinning, called-street closure, preflop menus, off-menu translation, engine street
    #    order, showdown-leaf settlement). These change the KEY SPACE or the money semantics the
    #    object was certified under; any mismatch is fail-closed.
    env_require: dict = stamp.get("env_require", {})
    bad = {k: (want, os.environ.get(k)) for k, want in env_require.items()
           if os.environ.get(k, "") != want}
    if bad:
        raise RuntimeError(
            f"tier-1 blueprint index {path}: live env does not match the certified serve config "
            f"(var: (required, actual)): {bad} — set the vars BEFORE importing pokerbot "
            "(pokerbot/arena/serve_env.py does this when POKERBOT_SERVE_STACK=tier1), or roll "
            "back with POKERBOT_SERVE_STACK=k160")

    # 2) Encoder abstraction stamp (import-latched): strict — raises on skew.
    expected = stamp.get("abstraction")
    if expected:
        from pokerbot.eval.lbr import check_blueprint_serve_env

        check_blueprint_serve_env(expected=expected, strict=True)

    snap = SortedSnapshotDir(path)
    want_rows = int(stamp.get("expected_rows", meta.get("n_strat", 0)) or 0)
    if want_rows and len(snap) != want_rows:
        raise RuntimeError(
            f"tier-1 blueprint index {path}: row count {len(snap):,} != certified "
            f"{want_rows:,} — truncated/corrupt index, refusing to serve")
    info = {"meta": meta, "serve_config": stamp, "n_rows": len(snap)}
    return snap, info
