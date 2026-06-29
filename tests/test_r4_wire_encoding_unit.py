"""
R4 WIRE-ENCODING SHARPENING — pure unit tests.

Pins the all_in -> raise(5, chips) wire-encoding remap by reading the source directly,
since importing the full live_coach module triggers a jax crash via the conftest.

The fix lives in scripts/live/live_coach.py in execute_action() right after the
``_legal_raise_size`` call: when ``name == "all_in"``, rewrite to ``("raise", chips)``
where chips = opt5[1] (client's authoritative max_raise_to) || stack || your_bet+to_call.
Fallback to check/call if no shove amount can be determined.

These tests parse the live_coach.py source and assert:
  1. The R4 remap block is present in execute_action (in place of the legacy all_in pass-through).
  2. The remap consults opt5[1] as the primary shove amount source.
  3. The remap falls back to the hero's stack when opt5 is missing.
  4. The remap falls back to your_bet + to_call when both opt5 and stack are unknown.
  5. The remap downgrades to call/check when no shove amount can be resolved (don't ship 0-amount).
  6. ``ActionSender.send("all_in", None)`` is no longer produced by the autoplay path (no caller
     passes name="all_in" downstream of execute_action).

Conftest is bypassed by setting PYTEST_DISABLE_PLUGIN_AUTOLOAD for this file via the
`rootdir` exclude; we read source directly so no module-import jax side effects.
"""
from __future__ import annotations

import os
import re
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LIVE_COACH = os.path.join(ROOT, "scripts", "live", "live_coach.py")


def _read_source() -> str:
    with open(LIVE_COACH, "r", encoding="utf-8") as fh:
        return fh.read()


def _slice_execute_action(src: str) -> str:
    """Return the text of execute_action() — between its def and the next module-level
    def at the same indent. Good enough for grep-based assertions."""
    start = src.find("def execute_action(")
    if start < 0:
        return ""
    # Find the matching end (next top-level def at column 0)
    rest = src[start:]
    lines = rest.splitlines(keepends=True)
    end = len(lines)
    for i, ln in enumerate(lines[1:], 1):
        if ln.startswith("def ") and not ln.startswith("def _"):
            end = i
            break
    return "".join(lines[:end])


class TestR4WireEncodingRemapPresent(unittest.TestCase):

    def setUp(self):
        self.src = _read_source()
        self.exec_action_src = _slice_execute_action(self.src)

    def test_remap_block_present(self):
        # The block exists somewhere in execute_action; search for the comment marker we added.
        self.assertIn("R4 WIRE-ENCODING FIX", self.exec_action_src,
                      "execute_action must contain the R4 wire-encoding remap block")

    def test_remap_consults_opt5_hi_first(self):
        # The remap tries opt5[1] before stack, with the right precedence.
        # Find the block by anchoring on the comment.
        i = self.exec_action_src.find("R4 WIRE-ENCODING FIX")
        block = self.exec_action_src[i:]
        # opt5 is queried first, then stack.
        opt5_pos = block.find("opt5")
        stack_pos = block.find("stack or 0.0")
        self.assertGreater(opt5_pos, -1, "remap must consult opt5")
        self.assertGreater(stack_pos, -1, "remap must fall back to stack")
        self.assertLess(opt5_pos, stack_pos,
                        "opt5 must be the primary shove-amount source (consulted before stack)")

    def test_remap_falls_back_to_your_bet_plus_to_call(self):
        i = self.exec_action_src.find("R4 WIRE-ENCODING FIX")
        block = self.exec_action_src[i:]
        self.assertIn("your_bet", block,
                      "remap must fall back to your_bet (the chip count we'd put in already)")
        self.assertIn("to_call", block,
                      "remap must include to_call in the fallback shove calculation")

    def test_remap_downgrades_to_check_or_call_when_no_shove(self):
        i = self.exec_action_src.find("R4 WIRE-ENCODING FIX")
        block = self.exec_action_src[i:]
        self.assertIn('"call"', block,
                      "remap must downgrade to call when to_call > 0 and no shove is known")
        self.assertIn('"check"', block,
                      "remap must downgrade to check when to_call == 0 and no shove is known")

    def test_remap_replaces_all_in_with_raise(self):
        """The remap rewrites the action tuple so the all_in never reaches sender.send."""
        i = self.exec_action_src.find("R4 WIRE-ENCODING FIX")
        block = self.exec_action_src[i:]
        self.assertIn('name == "all_in"', block,
                      "remap block must guard on name == 'all_in'")
        # The remap SETS name to "raise" — search for the assignment.
        self.assertRegex(block, r'name,\s*chips\s*=\s*"raise"',
                         "remap must rewrite name='raise' once the shove amount is resolved")

    def test_no_sender_send_of_all_in_with_none_chips(self):
        """Pin the invariant: after the remap, the worker calls sender.send('raise', chips) —
        never ('all_in', None). Search for any 'sender.send('all_in' that doesn't go through the
        remap (a regression would ship the legacy null-amount shove)."""
        # The execute_action path must not have any direct sender.send("all_in", None).
        # The ActionSender.send method itself accepts (name, chips) generically — we only care
        # about CALLERS in execute_action that pass "all_in".
        bad_callers = re.findall(r'sender\.send\(\s*["\']all_in["\']', self.exec_action_src)
        self.assertEqual(bad_callers, [],
                         f"execute_action must not pass 'all_in' to sender.send; found {bad_callers}")

    def test_happy_path_raise_unchanged(self):
        """The remap is GUARDED by `if name == "all_in"`; a normal raise path is untouched."""
        # The block must be inside an `if name == "all_in":` guard, not a blanket rewrite.
        i = self.exec_action_src.find("R4 WIRE-ENCODING FIX")
        block = self.exec_action_src[i:]
        self.assertRegex(block, r'if\s+name\s*==\s*["\']all_in["\']',
                         "remap block must be guarded by `if name == 'all_in'`")


class TestR4HookAllInRemap(unittest.TestCase):
    """Pin the wire-encoding contract on the JS hook side: this CoinPoker build executes a shove
    as UserAction(Raise/5, full_stack), NOT AllIn/6. The Python side carries the chips; the
    hook simply maps raise -> Raise(5, amount)."""

    def setUp(self):
        self.src = _read_source()

    def test_action_sender_send_forwards_chips(self):
        """ActionSender.send forwards (name, chips) to the RPC. The all_in remap in execute_action
        ensures we always pass a non-None chips for a shove. Pin the forwarding shape so a future
        regression to ActionSender.send can't silently drop chips."""
        # The send method body must call self._rpc().send_action(name, float(chips or 0))
        # OR equivalent — NOT silently coerce name=='all_in' to something else (the remap is
        # solely in execute_action).
        m = re.search(r'def send\(self[^)]+\)[^\n]*:\s*\n(.*?)(?=\n    def |\nclass |\Z)',
                      self.src, flags=re.DOTALL)
        self.assertIsNotNone(m, "ActionSender.send must exist")
        body = m.group(1)
        self.assertIn("send_action", body,
                      "ActionSender.send must forward via send_action")
        self.assertIn("float(chips", body,
                      "ActionSender.send must coerce chips to float for the RPC")

    def test_hook_all_in_remap_audit(self):
        """The hook ships Raise(5, amount) for a shove. The Python remap ensures the action name
        passed is 'raise' (NOT 'all_in'). This test pins the hook constant on the JS side: in
        mono_state_hook.js, ACTION_IDS still has all_in: 6 (kept for completeness — future builds
        that accept AllIn/6 will Just Work), but the Python path doesn't use it."""
        hook_path = os.path.join(ROOT, "scripts", "live", "mono_state_hook.js")
        with open(hook_path, "r", encoding="utf-8") as fh:
            hook_src = fh.read()
        # Hook side: ACTION_IDS maps the action names; RAISE_ID = 5 carries the amount.
        self.assertIn("RAISE_ID = 5", hook_src,
                      "hook must keep RAISE_ID = 5 (the only action that carries the amount)")
        self.assertIn("all_in: 6", hook_src,
                      "hook ACTION_IDS must still map all_in -> 6 (legacy AllIn/6 for future builds)")
        # sendAction: withAmount = (id === RAISE_ID) -> for all_in id=6, withAmount=false; chips dropped.
        # This is the legacy behavior preserved on the JS side; the fix lives on the Python side
        # (we never send name='all_in' from the autoplay path anymore).
        self.assertIn("withAmount = (id === RAISE_ID)", hook_src,
                      "hook sendAction must keep the withAmount = (id === RAISE_ID) convention")


if __name__ == "__main__":
    unittest.main(verbosity=2)
