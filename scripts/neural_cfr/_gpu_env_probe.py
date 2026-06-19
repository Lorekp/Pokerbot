"""Probe: does the GPU venv have the full neural-CFR training stack importable on CUDA?"""
import importlib, sys

for m in ["jax", "jaxlib", "flax", "optax", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print("OK  ", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("MISS", m, type(e).__name__, str(e)[:120])

import jax
print("devices:", jax.devices())
print("python:", sys.version.split()[0])

# Can the repo's training entrypoints import under this interpreter?
try:
    from pokerbot.training.cfr_env import leduc_env_spec, nlhe_env_spec  # noqa
    from pokerbot.training.neural_cfr import VRDeepPDCFRConfig  # noqa
    print("REPO import: OK")
except Exception as e:
    print("REPO import: FAIL", type(e).__name__, str(e)[:200])
