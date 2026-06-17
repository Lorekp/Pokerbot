"""pokerbot.training package.

The top-level convenience names below are exposed lazily (PEP 562 ``__getattr__``) so that
importing a JAX-only submodule (e.g. ``pokerbot.training.sd_cfr_nlhe``) does NOT eagerly pull in
torch via ppo/self_play/sft. Accessing any of the names still imports its module on demand, so the
public API (``from pokerbot.training import PPOConfig``) is unchanged. This lets the SD-CFR cloud
trainer run on a torch-free instance (``--no-sft-init``), where only JAX/numpy are installed.
"""

from typing import TYPE_CHECKING

__all__ = [
    "PPOConfig",
    "SFTConfig",
    "SelfPlayConfig",
    "train_self_play",
    "train_sft",
    "trinal_clip_ppo_loss",
]

_LAZY = {
    "PPOConfig": "pokerbot.training.ppo",
    "trinal_clip_ppo_loss": "pokerbot.training.ppo",
    "SelfPlayConfig": "pokerbot.training.self_play",
    "train_self_play": "pokerbot.training.self_play",
    "SFTConfig": "pokerbot.training.sft",
    "train_sft": "pokerbot.training.sft",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # static analysers / IDEs still see the real symbols
    from pokerbot.training.ppo import PPOConfig, trinal_clip_ppo_loss
    from pokerbot.training.self_play import SelfPlayConfig, train_self_play
    from pokerbot.training.sft import SFTConfig, train_sft
