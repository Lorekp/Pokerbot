"""Statistical opponent profiling (VPIP, PFR, AF, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpponentStats:
    hands: int = 0
    vpip: int = 0  # voluntarily put $ in pot preflop
    pfr: int = 0  # preflop raise
    three_bet: int = 0
    fold_to_cbet: int = 0
    cbet_faced: int = 0
    aggression_bets_raises: int = 0
    aggression_calls: int = 0
    wtsd: int = 0  # went to showdown
    saw_flop: int = 0

    @property
    def vpip_pct(self) -> float:
        return self.vpip / max(self.hands, 1)

    @property
    def pfr_pct(self) -> float:
        return self.pfr / max(self.hands, 1)

    @property
    def aggression_factor(self) -> float:
        return self.aggression_bets_raises / max(self.aggression_calls, 1)

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.fold_to_cbet / max(self.cbet_faced, 1)

    def archetype(self) -> str:
        if self.hands < 5:
            return "unknown"
        if self.vpip_pct < 0.15 and self.pfr_pct < 0.12:
            return "nit"
        if self.vpip_pct > 0.45 and self.aggression_factor < 1.5:
            return "calling_station"
        if self.vpip_pct > 0.35 and self.aggression_factor > 2.5:
            return "lag"
        if 0.18 <= self.vpip_pct <= 0.28 and self.pfr_pct >= 0.15:
            return "tag"
        return "unknown"


@dataclass
class OpponentTracker:
    """Track stats per villain seat/id."""

    profiles: dict[int, OpponentStats] = field(default_factory=dict)

    def get(self, player_id: int) -> OpponentStats:
        if player_id not in self.profiles:
            self.profiles[player_id] = OpponentStats()
        return self.profiles[player_id]

    def record_hand(
        self,
        player_id: int,
        *,
        vpip: bool = False,
        pfr: bool = False,
        three_bet: bool = False,
        saw_flop: bool = False,
        wtsd: bool = False,
        bet_or_raise: int = 0,
        calls: int = 0,
        fold_to_cbet: bool = False,
        cbet_faced: bool = False,
    ) -> None:
        s = self.get(player_id)
        s.hands += 1
        s.vpip += int(vpip)
        s.pfr += int(pfr)
        s.three_bet += int(three_bet)
        s.saw_flop += int(saw_flop)
        s.wtsd += int(wtsd)
        s.aggression_bets_raises += bet_or_raise
        s.aggression_calls += calls
        s.fold_to_cbet += int(fold_to_cbet)
        s.cbet_faced += int(cbet_faced)

    def confidence(self, player_id: int) -> float:
        """0-1 confidence based on sample size."""
        hands = self.get(player_id).hands
        return min(1.0, hands / 50.0)
