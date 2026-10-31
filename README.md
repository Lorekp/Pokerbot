# Pokerbot

**Poker simulation, strategy learning, and decisions with incomplete information.**

I built Pokerbot to explore how an agent can choose a strategy when it cannot see its opponents’ cards—and how to check whether that strategy is actually good.

This research codebase combines all-in-or-fold (AoF) and no-limit Texas Hold’em simulations, counterfactual regret minimization (CFR), learned policies, opponent models, and depth-limited search. It contains several generations of experiments, including approaches that failed or remained incomplete.

## Try one decision

Use Python **3.11+**, from the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -r requirements-aof.txt
python aof.py "AJo BTN 8bb" --mc 20000 --seed 42 --json
```

This evaluates pushing all-in with ace-jack offsuit from the button at an eight-big-blind stack. The checked example returns **ALL IN**, with an expected-value estimate and explicit warnings that its opponent ranges are hand-tuned. It does not account for rake or side pots in that configuration. These are model-dependent estimates, not a claim of profitable play.

Add `--explain` for a readable explanation. The historical `--nash` option name does not certify an equilibrium.

## What to explore

| Area | Code |
| --- | --- |
| Offline decisions, equity and confidence | [aof.py](aof.py), [aof_confidence.py](aof_confidence.py) |
| CFR policies and hidden-hand best responses | [aof_cfr/](aof_cfr/) |
| Reinforcement learning and self-play | [aof_rl/](aof_rl/) |
| No-limit simulation and search | [pokerbot/engine/](pokerbot/engine/), [pokerbot/search/](pokerbot/search/) |
| Opponent models and synthetic evaluation opponents | [pokerbot/opponent/](pokerbot/opponent/), [pokerbot/eval/](pokerbot/eval/) |
| Policy training and tournament experiments | [pokerbot/training/](pokerbot/training/), [pokerbot/tournament/](pokerbot/tournament/) |
| Native computational kernels | [tools/](tools/) |

The heads-up best-response implementation integrates possible opponent hands and the probability of reaching a decision. Search uses blueprint policies and continuation rollouts; training includes neural-policy and CFR experiments. These are implementations to inspect and experiment with, not independently certified solver-strength results.

## Validation and setup

```bash
python -m pytest tests/test_aof.py tests/test_aof_confidence.py tests/test_route_hu.py -q
```

The focused suite covers parsing, confidence labels, game economics, heads-up policy consistency and opponent routing. Its equity and policy fixtures are included under `data/aof/`; missing fixtures cause skips and must not be counted as passing checks.

For the wider Python project, install `python -m pip install -e ".[aof,dev]"`. Optional training dependencies are listed in [pyproject.toml](pyproject.toml). Advanced experiments may require generated data, model checkpoints, or separate native toolchains. The native wrappers do not bundle their upstream solver/evaluator dependencies. Full training, every historical experiment, and native builds are outside the small smoke suite’s coverage.

The fingerprint and battery Rust crates generate their lookup tables locally from the separately installed `phevaluator==0.5.3.1` dependency. Cargo runs the included generators before compilation; select the Python environment with `PYO3_PYTHON` or `PYTHON` if needed. Generated tables are ignored by Git. For explicit generation from the repository root:

```bash
python tools/fingerprint_native/gen_tables.py
python tools/battery_native/gen_phe_tables.py
```

Run research scripts from the repository root. Former machine-specific defaults now point to repository-relative `checkpoints/`, `data/external/` and `artifacts/` locations; supply the required inputs or override the script's path options. These optional input files are not bundled.

Some optional cloud launchers use an external Spotpick capacity helper, which is not included here. Set `SPOTPICK_ROOT` to its Python module directory or make it importable in the active environment; PowerShell babysitters use `SPOTPICK_SCRIPT` for its script path. Without that helper, use the existing explicit instance-type and price options/fallbacks. `POKERBOT_SEARCH_ROOTS` selects the optional captured research-root fixture for `native_exploit_equiv.py`; its default is `artifacts/search_roots.pkl`. Monitor scripts accept their existing root-directory environment overrides and otherwise use the script's repository root or the remote login directory.

## Scope and history

This public edition contains the research. Client automation, captures, human-mimicking behavior, bundled third-party files, and player datasets were removed from its history. Original commit authorship and dates are preserved; some commits are now empty, and older messages or research comments can refer to omitted integrations. Historical revisions are not promised to be independently runnable. A few computational modules retain legacy names such as `live_exploit`; they operate on supplied data rather than controlling a client.

The project builds on established methods and external work, including [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench), [AlphaNLHoldem](https://github.com/bupticybee/AlphaNLHoldem), [postflop-solver](https://github.com/b-inary/postflop-solver), and [OMPEval](https://github.com/zekyll/OMPEval). These are upstream references, not bundled distributions. Commit history records AI-assisted contributions. No blanket license is granted for the original project code; dependencies retain their own licenses.
