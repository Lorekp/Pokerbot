"""Economic models that wrap the chip-EV engine (rake, rakeback) without touching its purity.

The engine terminal (`np_engine.compute_returns_np` / `jax_engine.compute_returns`) stays pure chip-EV.
Rake and rakeback are applied *outside* it — in eval and (later) in the raked-equilibrium retrain — exactly
as the ICM overlay is meant to wrap the same single terminal point. See `rake.py`.
"""
