"""Safe GTO/exploit policy mixing (OX-Search inspired)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pokerbot.engine.actions import NUM_ACTIONS
from pokerbot.opponent.bayes_exploit import context_read
from pokerbot.opponent.context import context_key as _context_key
from pokerbot.opponent.stats import OpponentStats
from pokerbot.opponent.value_sizing import (
    bluff_suppression_logits,
    river_thin_value_logits,
    value_sizing_logits,
)

# board card count -> street index (preflop/flop/turn/river)
_BOARD_LEN_TO_STREET = {0: 0, 3: 1, 4: 2, 5: 3}


def _mixer_context_key(board, legal_mask, n_prior_aggr: int | None = None) -> str:
    """Per-context key from what the policy path has: ``board`` (→ street + texture) and either the
    exact per-street aggression count (``n_prior_aggr``, threaded from the decision path) or, when that
    is unavailable, the legal mask. Same pokerbot.opponent.context spec the search and offline ingestion
    use (train/serve-skew firewall).

    When ``n_prior_aggr`` is supplied the key is EXACT (full preflop 3bet/4bet depth). When it is None
    we fall back to the CHECK-legal binary (CHECK legal ⟺ unopened; FOLD is legal even when checking is,
    so it can't signal facing) — a strict COARSENING (only ``unopened``/``facing_bet``), never a key
    that conflicts with the recorded profile."""
    cards = [str(c) for c in (board or [])]
    street = _BOARD_LEN_TO_STREET.get(len(cards), 0)
    if n_prior_aggr is not None:
        return _context_key(street, cards, int(n_prior_aggr))
    check_legal = False
    if legal_mask is not None:
        arr = legal_mask.detach().cpu().numpy() if hasattr(legal_mask, "detach") else np.asarray(legal_mask)
        arr = arr.reshape(-1)
        check_legal = bool(arr.size > 1 and arr[1] > 0)
    return _context_key(street, cards, 0 if check_legal else 1)


def _context_or_archetype_logits(stats, hand_strength, board, legal_mask,
                                 n_prior_aggr: int | None = None) -> np.ndarray:
    """OUR exploit logit adjustment, preferring the per-context Bayesian read (bayes_exploit) when
    that context has data, else the discrete-archetype ``exploit_adjustment`` (cold-start fallback).

    On top of the frequency read we add two HUD-driven SIZE/FREQUENCY overlays: ``value_sizing_logits``
    escalates HOW BIG we bet value vs a sticky/weak-showdown station, and ``bluff_suppression_logits``
    (the symmetric air-hand term) slashes HOW OFTEN we bluff a non-folder. Both are value/air-gated,
    confidence-gated, additive, and bounded, so the mixer α-cap and LAG/RNR firewall still clamp them.
    Env kill-switches POKERBOT_VALUE_SIZING=0 / POKERBOT_BLUFF_SUPPRESS=0."""
    read = context_read(stats, _mixer_context_key(board, legal_mask, n_prior_aggr))
    base = read.mixer_logits(hand_strength) if read is not None else exploit_adjustment(stats, hand_strength)
    return (base + value_sizing_logits(stats, hand_strength)
            + bluff_suppression_logits(stats, hand_strength)
            + river_thin_value_logits(stats, hand_strength, board))

USE_BELIEF_BR = True

# Default until `scripts/calibrate_mixer_threshold.py` writes checkpoints/mixer_threshold.json.
# PROVISIONAL / UNCALIBRATED for live deployment — see docs/phase3-runbook.md § Mixer confidence.
_DEFAULT_BELIEF_CONFIDENCE_THRESHOLD = 0.35


def _load_calibrated_threshold() -> float:
    path = Path("checkpoints/mixer_threshold.json")
    if not path.exists():
        return _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("chosen_threshold", _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD))
    except (OSError, ValueError, TypeError):
        return _DEFAULT_BELIEF_CONFIDENCE_THRESHOLD


BELIEF_CONFIDENCE_THRESHOLD = _load_calibrated_threshold()
TRANSFORMER_LOGIT_WEIGHT = 1.0
STATS_FALLBACK_WEIGHT = 0.2


def exploit_adjustment(stats: OpponentStats, hand_strength: float | None = None) -> np.ndarray:
    """
    Return logit adjustments for each of OUR action indices, biasing our own policy toward the
    +EV response to the opponent's archetype.
    Indices: 0=fold, 1=check, 2=call, 3..NUM_ACTIONS-2 = bet/raise size buckets (ascending),
    NUM_ACTIONS-1 = all-in.

    ``hand_strength`` is hero's equity proxy in [0, 1] (None when the caller can't supply it). It
    matters for the calling-station branch: against a station (calls too much, folds too little)
    the correct exploit is to VALUE-BET MORE with strong holdings and BLUFF LESS with weak ones —
    *not* to cut betting uniformly, which forfeits value. Real-population data confirms loose-passive
    stations dominate (median VPIP ~48%, AF ~0.65), so getting this direction right is the highest-
    value lever. When ``hand_strength`` is None we apply a value-neutral tilt that never suppresses
    betting on average.
    """
    adj = np.zeros(NUM_ACTIONS, dtype=np.float32)
    arch = stats.archetype()
    bet_lo = 3                       # first bet/raise bucket
    big_lo = max(bet_lo, NUM_ACTIONS - 4)  # largest sizings (~>=1x pot) + all-in

    if arch == "nit":
        adj[bet_lo:] += 0.5  # folds too much -> apply pressure across all sizings
        adj[0] -= 0.3
    elif arch == "calling_station":
        # Station pays off value and won't fold to bluffs. Boost calls (cheaply realize equity
        # vs their aggression), then scale OUR bet/raise frequency by hand strength: strong hand
        # -> value-bet MORE and bigger (they call anyway); weak hand -> bluff LESS.
        adj[2] += 0.2
        h = 0.55 if hand_strength is None else float(np.clip(hand_strength, 0.0, 1.0))
        value = (h - 0.5) * 2.0  # in [-1, 1]: >0 value hand, <0 air
        adj[bet_lo:] += 0.6 * value
        if value > 0.0:
            adj[big_lo:] += 0.4 * value  # extract more from a station with larger sizes
    elif arch == "lag":
        adj[0] += 0.3
        adj[2] += 0.2
    elif arch == "tag":
        adj[3:6] += 0.2
    return adj


def safe_mix_policies(
    gto_probs: np.ndarray,
    exploit_logits: np.ndarray,
    alpha: float,
    max_alpha: float = 0.7,
) -> np.ndarray:
    """
    π_final = (1-α)·π_GTO + α·π_exploit
    α gated by confidence; renormalize.
    """
    alpha = float(np.clip(alpha, 0.0, max_alpha))
    exploit_probs = np.exp(exploit_logits - exploit_logits.max())
    exploit_probs /= exploit_probs.sum() + 1e-8
    mixed = (1 - alpha) * gto_probs + alpha * exploit_probs
    mixed = np.clip(mixed, 1e-8, 1.0)
    return mixed / mixed.sum()


def _opponent_seat_index(actor_seat: int, hero_seat: int) -> int:
    """Map absolute seat to opponent index in [0, 4] relative to hero."""
    from pokerbot.opponent.sequence_encoder import opponent_index_for_seat

    return opponent_index_for_seat(actor_seat, hero_seat)


def _belief_confidence(belief_logits) -> float:
    import torch

    belief = torch.softmax(belief_logits, dim=-1)
    entropy = -(belief * torch.log(belief + 1e-8)).sum().item()
    max_entropy = np.log(belief.shape[-1])
    return float(1.0 - entropy / max(max_entropy, 1e-8))


def _stats_confidence(stats) -> float:
    """Sample-size confidence min(1, hands/50) from the stats object ACTUALLY used for the exploit.

    H7.1 fix: the blend alpha must track the same stats the exploit logits came from. When the read is
    a persistent-DB ``opponent_profile`` (a player known only from the DB), the in-memory tracker is
    empty for that id, so gating alpha on ``tracker.confidence(id)`` silently zeroed DB-driven
    exploitation — defeating the "remember every player" goal. Sourcing it from ``stats.hands`` is
    BIT-IDENTICAL for the tracker path (there ``stats = tracker.get(id)``, and
    ``tracker.confidence(id) == min(1, tracker.get(id).hands/50)``) while making the DB profile drive
    the deviation.

    HUD floor: a server-HUD prior (whole-history vpip/pfr/wtsd/wsd/foldToCbet/...) is high-confidence
    even with few LOCALLY observed hands. Without this floor a station known only from the CoinPoker HUD
    (hands < ~5 locally) is alpha-zeroed -> the value-sizing lever silently doesn't fire vs the exact
    target it was built for (see population-hud-analysis / ev-lever-implementation)."""
    conf = min(1.0, float(getattr(stats, "hands", 0)) / 50.0)
    hud = getattr(stats, "hud", None)
    if isinstance(hud, dict) and len(hud) >= 3:
        # FIX B2: ramp the HUD floor instead of a flat 0.6, so 0.6 is reached only after ~5 corroborating
        # LOCAL hands (0.1 + 0.1*hands). A 1-hand HUD read -> ~0.2, not full exploit trust on hand one.
        hands = float(getattr(stats, "hands", 0))
        conf = max(conf, min(0.6, 0.1 + 0.1 * hands))
    return conf


class HybridPolicy:
    """Combines GTO policy network with MADT opponent-aware adjustments."""

    def __init__(self, model, opponent_tracker, opponent_transformer=None, device="cpu"):
        self.model = model
        self.tracker = opponent_tracker
        self.opponent_transformer = opponent_transformer
        self.device = device

    def get_probs(
        self,
        weva_ids,
        cards,
        actions,
        position,
        legal_mask,
        opponent_id: int,
        transformer_out: dict | None = None,
        actor_seat: int | None = None,
        hero_seat: int = 0,
        board: list | None = None,
        trajectory=None,
        opponent_profile=None,
        hand_strength: float | None = None,
        n_prior_aggr: int | None = None,
        gift_bonus: float = 0.0,
        exploit: bool = True,
        exploit_strength: float = 1.0,
        base_probs_override: np.ndarray | None = None,
    ) -> np.ndarray:
        # Pluggable GTO base (task (b)): when the caller supplies an 11-vector base policy (e.g. the
        # CFR blueprint / depth-limited search), the read->mixer blends against IT instead of the SFT
        # net forward pass. base_probs_override=None ⇒ the validated SFT path, BIT-IDENTICAL (no
        # torch import side effects either). The exploit read, alpha gating, and legal masking below
        # are base-agnostic — they operate on whatever ``gto`` is. This is the seam the table-toughness
        # router uses to fall back to a strong GTO base on tough tables while keeping the SFT base
        # (which extracts more from weak fields) as the default value vehicle.
        if base_probs_override is not None:
            gto = np.asarray(base_probs_override, dtype=np.float64).reshape(-1)
        else:
            import torch

            with torch.no_grad():
                out = self.model(
                    weva_ids.unsqueeze(0) if weva_ids.dim() == 0 else weva_ids,
                    actions.unsqueeze(0) if actions.dim() == 1 else actions,
                    position.unsqueeze(0) if position.dim() == 1 else position,
                    legal_mask.unsqueeze(0) if legal_mask.dim() == 1 else legal_mask,
                    cards=cards.unsqueeze(0) if cards.dim() == 3 else cards,
                )
                gto = out.probs.squeeze(0).cpu().numpy()

        # Clean exploit toggle (the A/B OFF arm = PURE GTO base). Without this, opponent_profile=None
        # still mixes off the in-memory tracker, so the battery would measure DB-profile-vs-tracker
        # rather than exploit-vs-GTO. exploit=False short-circuits to the masked GTO policy.
        if not exploit:
            final = np.asarray(gto, dtype=np.float64)
            if legal_mask is not None:
                lm = legal_mask.cpu().numpy().astype(bool)
                if lm.ndim > 1:
                    lm = lm.squeeze(0)
                final = final * lm
                final = final / (final.sum() + 1e-8)
            return final

        # Prefer a persistent-DB profile when the live driver supplies one (seam for the future
        # exploitation build); default None => the in-memory tracker, i.e. unchanged behavior.
        stats = opponent_profile if opponent_profile is not None else self.tracker.get(opponent_id)
        transformer_confidence = 0.0
        exploit_logits = np.log(gto + 1e-8)

        if transformer_out is not None and actor_seat is not None:
            policy_logits = transformer_out.get("policy_logits")
            belief_logits = transformer_out.get("belief_logits")
            if policy_logits is not None:
                opp_idx = _opponent_seat_index(actor_seat, hero_seat)
                opp_idx = min(max(opp_idx, 0), 4)
                tf_logits = policy_logits[0, opp_idx].cpu().numpy()
                if belief_logits is not None:
                    transformer_confidence = _belief_confidence(belief_logits[0, opp_idx])
                if transformer_confidence >= BELIEF_CONFIDENCE_THRESHOLD:
                    if USE_BELIEF_BR and belief_logits is not None:
                        from pokerbot.opponent.belief_br import belief_br_from_transformer_output

                        exploit_logits = belief_br_from_transformer_output(
                            belief_logits,
                            policy_logits,
                            opp_idx,
                            transformer=self.opponent_transformer,
                            trajectory=trajectory,
                            hero_seat=hero_seat,
                            device=self.device,
                            board=board,
                        )
                    else:
                        exploit_logits = TRANSFORMER_LOGIT_WEIGHT * tf_logits
                    exploit_logits = exploit_logits + STATS_FALLBACK_WEIGHT * _context_or_archetype_logits(
                        stats, hand_strength, board, legal_mask, n_prior_aggr
                    )
                else:
                    exploit_logits = exploit_logits + _context_or_archetype_logits(
                        stats, hand_strength, board, legal_mask, n_prior_aggr
                    )
                    exploit_logits = exploit_logits + 0.35 * tf_logits
        else:
            exploit_logits = exploit_logits + _context_or_archetype_logits(
                stats, hand_strength, board, legal_mask, n_prior_aggr
            )

        # R4 gift-modulated bounded-ε: a banked-gift budget (threaded from the agent's GiftRegister)
        # relaxes the alpha cap for this opponent. gift_bonus=0 ⇒ cap=0.7 ⇒ unchanged behavior.
        from pokerbot.opponent.safety_budget import ALPHA_CEILING

        cap = min(ALPHA_CEILING, 0.7 + max(0.0, float(gift_bonus)))
        if transformer_confidence >= BELIEF_CONFIDENCE_THRESHOLD:
            alpha = min(cap, 0.35 + transformer_confidence * 0.35)
        else:
            # H7.1: gate alpha on the SAME stats the exploit came from (profile or tracker), so a
            # persistent-DB profile actually drives deviation. Bit-identical for the tracker path.
            alpha = _stats_confidence(stats) * 0.5
            alpha += transformer_confidence * 0.25
            if stats.archetype() == "unknown" and transformer_confidence < 0.1:
                alpha = 0.0

        # Skill-tier exploitation dial: scale the mixer alpha by exploit_strength in [0,1]. 1.0 ⇒ the
        # full validated overlay (unchanged); 0.0 ⇒ pure GTO base (== exploit disabled). Maps
        # ADAPT[tier].exploit_strength so advanced exploits softly and superhuman fully.
        alpha *= float(max(0.0, min(1.0, exploit_strength)))
        final = safe_mix_policies(gto, exploit_logits, alpha, max_alpha=cap)
        if legal_mask is not None:
            lm = legal_mask.cpu().numpy().astype(bool)
            if lm.ndim > 1:
                lm = lm.squeeze(0)
            final = final * lm
            final /= final.sum() + 1e-8
        return final
