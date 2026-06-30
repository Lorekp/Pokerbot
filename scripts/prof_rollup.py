"""Roll up a cProfile .prof into self-time (tottime) buckets by logical component, for attribution."""
import pstats, sys, collections

st = pstats.Stats(sys.argv[1])
total = st.total_tt
buckets = collections.OrderedDict([
    ("hand-eval (score_five/_straight_high/evaluate_seven/player_showdown_score)", []),
    ("showdown settlement (compute_returns/award_layer/_split_consume/_odd_chip/_eligible)", []),
    ("apply_action_np (engine step, excl showdown)", []),
    ("legal_action_mask_np", []),
    ("RNG (_threefry/_rotl32/sample_chance/_deal_n_cards)", []),
    ("blueprint lookup (fingerprint/betting_abstraction_key/policy/blueprint/bias_continuation/continuation_set)", []),
    ("infoset/leaf keys (_infoset_key/_leaf_key/_public_seed/canonicalize)", []),
    ("CFR recursion glue (_cfv/_opponent_strategy/_regret_match/_cached_blueprint/_playout/_sample_action)", []),
    ("numpy reductions/wrappers (reduce/any/sum/_wrapreduction/asarray/astype/copy)", []),
    ("namedtuple _replace/_make/__new__", []),
])
def classify(fn):
    f = fn.lower()
    name = fn
    if any(k in f for k in ("score_five", "_straight_high", "evaluate_seven", "player_showdown_score", "_classify", "_rank_counts")):
        return 0
    if any(k in f for k in ("compute_returns", "award_layer", "_split_consume", "_odd_chip", "_eligible_to_win", "_pot_for_sizing")):
        return 1
    if "apply_action_np" in f:
        return 2
    if "legal_action_mask_np" in f:
        return 3
    if any(k in f for k in ("_threefry", "_rotl32", "sample_chance", "_deal_n_cards", "default_rng", "random")):
        return 4
    if any(k in f for k in ("fingerprint", "betting_abstraction_key", "lbr.py:355", "blueprint", "bias_continuation", "continuation_set", "_renorm_legal", "_apply_abstraction")):
        return 5
    if any(k in f for k in ("_infoset_key", "_leaf_key", "_public_seed", "canonicalize", "blake2b")):
        return 6
    if any(k in f for k in ("_cfv", "_opponent_strategy", "_regret_match", "_cached_blueprint", "_playout", "_sample_action", "_cont_and_mask", "_reshuffle", "continuation_leaf_value", "_leaf_value", "_build_leaf")):
        return 7
    if any(k in f for k in ("reduce", "fromnumeric", "_wrapreduction", "numpy.asarray", "'astype'", "'copy'", "'sum'", "method 'any'")):
        return 8
    if any(k in f for k in ("_replace", "_make", "__new__", "collections")):
        return 9
    return None

keys = list(buckets.keys())
other = 0.0
for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
    label = f"{func} {fname}:{line}"
    b = classify(label)
    if b is None:
        other += tt
    else:
        buckets[keys[b]].append(tt)

print(f"TOTAL tottime = {total:.1f}s  (cProfile-inflated wall)")
print("Self-time (tottime) attribution by component:")
acc = 0.0
for k in keys:
    s = sum(buckets[k]); acc += s
    print(f"  {100*s/total:6.1f}%  {s:8.1f}s  {k}")
print(f"  {100*other/total:6.1f}%  {other:8.1f}s  (other/unclassified)")
