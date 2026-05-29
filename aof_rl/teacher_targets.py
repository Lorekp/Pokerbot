"""Teacher distillation targets for PPO league (Q, probs, action)."""

from __future__ import annotations

import numpy as np

from aof import AdvisorTeacher
from aof_rl.env import ACTION_ALL_IN, ACTION_FOLD

NUM_ACTIONS = 2


def teacher_q_bb(teacher: AdvisorTeacher) -> np.ndarray:
    """Per-action chip EV in bb: index 0=FOLD, 1=ALL_IN."""
    q = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if teacher.action == "ALL IN":
        q[ACTION_ALL_IN] = float(teacher.ev_primary)
        q[ACTION_FOLD] = float(teacher.ev_alt)
    else:
        q[ACTION_FOLD] = float(teacher.ev_primary)
        q[ACTION_ALL_IN] = float(teacher.ev_alt)
    return q


def teacher_probs(teacher: AdvisorTeacher) -> np.ndarray:
    """Soft targets [p_fold, p_all_in] from teacher EVs (Boltzmann-lite)."""
    q = teacher_q_bb(teacher)
    margin = float(np.max(q) - np.min(q))
    if margin < 1e-6:
        p = np.array([0.5, 0.5], dtype=np.float32)
    else:
        # Sharp preference toward better action without full softmax temperature tuning.
        best = int(np.argmax(q))
        p = np.array([0.05, 0.05], dtype=np.float32)
        p[best] = 0.9
    return p


def teacher_action_index(teacher: AdvisorTeacher) -> int:
    return int(teacher.action_index)
