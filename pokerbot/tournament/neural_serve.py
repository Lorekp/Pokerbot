"""Skew-safe 7-seat neural serve bridge — the ONLY place the tournament eval consults ``seven_v1.pkl``.

This is the tournament analogue of ``pokerbot/api/hu_neural_endgame.py``: a lazily-loaded, fail-safe
serve wrapper around the trained 7-max ``AverageStrategyNet`` (``checkpoints/sd_cfr_7max_neural/
seven_v1.pkl``, produced by ``cfr_env.nlhe_7max_env_spec``). It is the single seam through which
``tournament.lbr_icm`` / ``regime_battery`` / ``mtt_lifecycle`` (other builders' modules) consult the
net; nothing else touches the checkpoint.

SEAT-SKEW IS THE #1 CORRECTNESS RISK (the load-bearing invariant)
----------------------------------------------------------------
``encode_info_state`` one-hots the ABSOLUTE seat and button index, sized to the table width
(``one_hot(seat, seat_status.shape[0])`` / ``one_hot(button, ...)`` — jax_engine.py:649-674). The 7-max
net was trained with ALL SEVEN seats live in NATURAL absolute order (occupancy ``[True]*7``, button
rotating over every seat — cfr_env.py:204-257). So a genuinely-7-live root needs **NO permutation**: the
served absolute indices already coincide with the trained ones (IDENTITY remap) — UNLIKE the HU net,
which pins its live pair at {3,5} and therefore always permutes. The ``<7-live`` sub-table case is
handled by ``seat_remap.place_live_seats`` (scatter holes onto chosen trained absolute indices). A
round-trip ``encode_info_state`` byte-identity test (``tests/test_seven_neural_serve.py``) proves the
bridge reproduces the native 7-seat node bit-for-bit, so the served encoder is byte-identical to what the
net saw during training — no silent encoder skew (the class of bug that turns a trained net into garbage).

The ``POKERBOT_INFO_SUITS`` skew guard is copied VERBATIM from ``hu_neural_endgame.py:156-164`` /
``hu_lbr_neural.make_neural_policy:57-64``: if the checkpoint's ``info_suits`` flag disagrees with the
serving process's ``jax_engine._INFO_SUITS``, the load FAILS CLOSED (the suit-feature tail changes the
info-state layout, so a mismatch silently misaligns every feature after it).

NOTE: ``encode_history_state``'s ``NUM_PLAYERS=6`` hardcode (jax_engine.py:684) is a TRAINING-side
history encoder; it does NOT touch the served ``AverageStrategyNet`` (which consumes only
``encode_info_state`` + ``legal_action_mask``), so it is irrelevant to this serve path.

FAIL-SAFE: this module NEVER raises into the serve path. A missing checkpoint, a load error, an
encoder-skew assertion, or a served distribution that collapses to (near-)zero legal mass all return
``None`` so the caller transparently falls back to a net-free anchor (always-fold / always-jam).

Smoke (WSL Debian) — CPU jax is fastest for single-example serves; net-free anchors need no GPU:
  source ~/jaxgpu-venv/bin/activate
  cd .
  PYTHONPATH=. POKERBOT_INFO_SUITS=1 JAX_PLATFORMS=cpu \
    python -m pytest tests/test_seven_neural_serve.py -q
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

ENV_CKPT = "POKERBOT_7MAX_NEURAL_CKPT"
_DEFAULT_SEVEN_CKPT = "checkpoints/sd_cfr_7max_neural/seven_v1.pkl"


class SevenMaxNeuralPolicy:
    """Lazily-loaded, fail-safe serve wrapper around the trained 7-max ``AverageStrategyNet``.

    ``policy(env_np, seat)`` returns an ``NUM_ACTIONS``-vector of action probabilities for the given
    (already trained-layout) numpy ``EnvState`` and ABSOLUTE seat, or ``None`` to signal the caller to
    fall back. The env MUST already be in the trained 7-seat absolute layout: for a full 7-live table
    that is the natural env (identity); for a ``<7-live`` sub-table, pass it through
    ``seat_remap.place_live_seats`` first and use the returned trained seat index.
    """

    def __init__(self, checkpoint: str | None = None):
        self.checkpoint = checkpoint or os.environ.get(ENV_CKPT, _DEFAULT_SEVEN_CKPT)
        self._serve = None       # jitted (state, seat) -> (probs, mask)
        self._meta: dict | None = None
        self._disabled = False   # set on an unrecoverable load/skew error -> permanent fallback
        self._fields = None      # EnvState._fields, cached after load

    # -- loading -------------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        if self._serve is not None:
            return True
        if self._disabled:
            return False
        try:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(self.checkpoint)
            import jax
            from pokerbot.engine.jax_engine import (
                EnvState, encode_info_state, legal_action_mask,
            )
            from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params

            payload = load_network_params(self.checkpoint)
            hidden = int(payload.get("hidden_dim", 256))
            params = payload["average_strategy_params"]
            net = AverageStrategyNet(hidden)

            # Skew guard (VERBATIM from hu_neural_endgame.py:156-164): the served encode_info_state
            # suit-feature layout MUST match training, or every feature after the suit tail misaligns.
            trained_suits = payload.get("info_suits")
            if trained_suits is not None:
                import pokerbot.engine.jax_engine as _je
                if bool(trained_suits) != bool(_je._INFO_SUITS):
                    raise RuntimeError(
                        f"info-state encoding skew: checkpoint POKERBOT_INFO_SUITS="
                        f"{int(bool(trained_suits))} but serving with {int(bool(_je._INFO_SUITS))}; "
                        f"set POKERBOT_INFO_SUITS to match BEFORE importing pokerbot.")

            @jax.jit
            def _serve(state, seat):
                info = encode_info_state(state, seat)
                mask = legal_action_mask(state, seat)
                probs = net.apply(params, info, mask)
                return probs * mask, mask

            self._serve = _serve
            self._fields = EnvState._fields
            self._meta = {"hidden_dim": hidden, "iteration": payload.get("iteration"),
                          "info_suits": payload.get("info_suits"), "checkpoint": self.checkpoint}
            _log.info("[seven_neural] loaded 7-max serve net %s meta=%s", self.checkpoint, self._meta)
            return True
        except (KeyError, ValueError, IndexError, OSError, AttributeError, RuntimeError):  # narrow: load path - any encoding/IO/class shape failure
            _log.exception("[seven_neural] failed to load %s; 7-max neural serve DISABLED this session",
                           self.checkpoint)
            self._disabled = True
            return False

    @property
    def meta(self) -> dict | None:
        return self._meta

    # -- serve ---------------------------------------------------------------
    def policy(self, env_np, seat: int) -> np.ndarray | None:
        """Serve the 7-max net for a trained-layout numpy ``EnvState``. Returns probs (len NUM_ACTIONS)
        or ``None`` (fall back). Never raises; a degenerate (<=1e-9 total legal mass) serve -> ``None``
        rather than a forced-uniform masquerading as a real decision."""
        if not self._ensure_loaded():
            return None
        try:
            import jax.numpy as jnp

            from pokerbot.engine.actions import NUM_ACTIONS

            state = type(env_np)(**{f: jnp.asarray(getattr(env_np, f)) for f in self._fields})
            probs, mask = self._serve(state, jnp.int32(int(seat)))
            p = np.asarray(probs, dtype=np.float64)
            m = np.asarray(mask, dtype=bool)
            p = p * m
            s = p.sum()
            if s <= 1e-9:
                # Net put no mass on any legal action -> degenerate serve; fall back rather than emit a
                # forced-uniform that would masquerade as a real decision.
                _log.warning("[seven_neural] degenerate serve (mass=%.2e), falling back", s)
                return None
            out = p / s
            if out.shape[0] != NUM_ACTIONS:
                return None
            return out
        except (KeyError, ValueError, IndexError, OSError, AttributeError, RuntimeError):  # narrow: serve path
            _log.exception("[seven_neural] serve failed; falling back")
            return None


def make_seven_neural_policy(checkpoint: str | None = None):
    """Return ``(policy_fn, meta)`` exposing ``policy_fn(env_np, seat) -> np.ndarray`` probs.

    This is the protocol that ``lbr_icm`` / ``regime_battery`` / ``mtt_lifecycle`` consume (mirroring
    ``hu_lbr_neural.make_neural_policy:49-88``). The field-by-field ``EnvState`` rebuild is already
    7-wide-generic (it iterates ``EnvState._fields``), so no HU-specific assumption leaks in.

    Unlike :class:`SevenMaxNeuralPolicy`.policy (which returns ``None`` on a degenerate serve so a caller
    can fall back to an anchor), ``policy_fn`` follows the LBR-harness contract and returns a forced
    legal-uniform on a degenerate serve (so an exploitability/regime measurement always gets a valid
    distribution and merely records the forced-uniform as a flat decision). This MIRRORS
    ``hu_lbr_neural.make_neural_policy`` exactly. Loading still FAILS LOUD here (raises) — the harness
    constructs the policy once up front, where a bad checkpoint should surface immediately, rather than
    silently measuring nothing.
    """
    import jax
    import jax.numpy as jnp

    from pokerbot.engine.jax_engine import (
        EnvState, encode_info_state, legal_action_mask,
    )
    from pokerbot.training.neural_cfr import AverageStrategyNet, load_network_params

    ckpt = checkpoint or os.environ.get(ENV_CKPT, _DEFAULT_SEVEN_CKPT)
    payload = load_network_params(ckpt)
    hidden = int(payload.get("hidden_dim", 256))
    params = payload["average_strategy_params"]
    net = AverageStrategyNet(hidden)

    # Skew guard (VERBATIM from hu_lbr_neural.make_neural_policy:57-64).
    trained_suits = payload.get("info_suits")
    if trained_suits is not None:
        import pokerbot.engine.jax_engine as _je
        if bool(trained_suits) != bool(_je._INFO_SUITS):
            raise RuntimeError(
                f"info-state encoding skew: checkpoint trained with POKERBOT_INFO_SUITS="
                f"{int(bool(trained_suits))} but serving with {int(bool(_je._INFO_SUITS))}. "
                f"Set POKERBOT_INFO_SUITS to match BEFORE importing pokerbot.")

    @jax.jit
    def _serve(state, seat):
        info = encode_info_state(state, seat)
        mask = legal_action_mask(state, seat)
        probs = net.apply(params, info, mask)
        return probs * mask, mask

    fields = EnvState._fields

    def policy_fn(env_np, seat: int) -> np.ndarray:
        state = EnvState(**{f: jnp.asarray(getattr(env_np, f)) for f in fields})
        probs, mask = _serve(state, jnp.int32(int(seat)))
        p = np.asarray(probs, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        p = p * m
        s = p.sum()
        if s > 1e-12:
            return p / s
        mm = m.astype(np.float64)
        return mm / max(mm.sum(), 1.0)

    return policy_fn, {"hidden_dim": hidden, "iteration": payload.get("iteration"),
                       "info_suits": payload.get("info_suits"), "checkpoint": ckpt}
