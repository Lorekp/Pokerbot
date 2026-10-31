"""Lazy-import registry of external opponent policy factories.

Usage:
    from pokerbot.eval.external_bots.registry import get_opponent_policy
    policy_fn = get_opponent_policy('fish')
    policy_fn = get_opponent_policy('honest')
    policy_fn = get_opponent_policy('deepbot', model_path='/path/to/model')
"""
from __future__ import annotations

import importlib
from typing import Any

# Registry: name -> (module_path, factory_function_name)
OPPONENTS: dict[str, tuple[str, str]] = {
    "fold_man": ("pokerbot.eval.external_bots.pypoker_adapter", "make_fold_man_policy"),
    "random":   ("pokerbot.eval.external_bots.pypoker_adapter", "make_random_policy"),
    "fish":     ("pokerbot.eval.external_bots.pypoker_adapter", "make_fish_policy"),
    "honest":   ("pokerbot.eval.external_bots.pypoker_adapter", "make_honest_policy"),
    "deepbot":  ("pokerbot.eval.external_bots.deepbot_adapter",  "make_deepbot_policy"),
}


def get_opponent_policy(name: str, **kwargs: Any):
    """Return a PolicyFn for the named opponent.

    Parameters
    ----------
    name   : one of the keys in OPPONENTS
    kwargs : forwarded verbatim to the factory function

    Returns
    -------
    PolicyFn  — (env, seat) -> np.ndarray[11] probability distribution

    Raises
    ------
    KeyError          if name is not in OPPONENTS
    ImportError       if the adapter module is missing or its dependency is unavailable
    AttributeError    if the factory function is not found in the module
    """
    if name not in OPPONENTS:
        available = sorted(OPPONENTS)
        raise KeyError(
            f"Unknown opponent {name!r}. Available: {available}"
        )
    module_path, factory_name = OPPONENTS[name]
    mod = importlib.import_module(module_path)
    factory = getattr(mod, factory_name)
    return factory(**kwargs)


def list_opponents() -> list[str]:
    """Return sorted list of registered opponent names."""
    return sorted(OPPONENTS)
