"""AoF evaluation-metrics suite.

Measures four aspects about the AoF engine, each as a gated cell in a versioned report:

  1. GTO base quality        — exact/MC legal NashConv per seat count and depth (:mod:`nash_gates`).
  2. Exploitation uplift     — bb/100 vs leaky/mixed/GTO fields with honest multi-seed CIs,
                               CRN-paired arms, duplicate seat rotation, and an AIVAT decision-term
                               control variate (:mod:`paired_arena`).
  3. Adaptation speed        — uplift as a function of hands observed; ``hands_to_half_uplift``
                               (:mod:`coldstart`).
  4. Safety                  — adapting adversaries (gear switcher, counter-exploiter) and the
                               Ganzfried–Sandholm exploitability budget (:mod:`adversary`).

Run the whole suite with ``python scripts/run_aof_eval_suite.py``; the artifact lands in
``runs/aof_eval/<stamp>_<gitsha>/metrics.json`` (:mod:`result`). Every cell reports gross AND
net-of-fee bb/100 — CoinPoker AoF charges a flat 0.12BB/hand, a −12 bb/100 floor any live edge
must clear.
"""

from aof_cfr.evalsuite.result import AoFEvalCell, write_report  # noqa: F401
