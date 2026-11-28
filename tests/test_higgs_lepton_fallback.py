import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "configs/Run3_test_20261127_higgs_lepton_fallback/higgs_labels.py"
spec = importlib.util.spec_from_file_location("higgs_lepton_fallback", path)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def tree(n=1):
    known = np.zeros((n, 12), dtype="i2")
    return dict(ancestryHiggs=np.full((n, 12), -1, "i1"), relationHiggs=np.full((n, 12), -1, "i1"),
                higgsPairKnown=known, higgsPairSame=known.copy(), valid=np.ones((n, 12), bool),
                pdg=np.broadcast_to([11, -13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0], (n, 12)).copy())


class TestHiggsLeptonFallback(unittest.TestCase):
    def test_single_multiple_zero_and_untouched_objects(self):
        a = tree(3)
        a["valid"][2, 0] = False
        a["pdg"][2, 1] = 0
        masks = np.array([[2, 2, 1], [-1, 0, 1], [1, 1, 0]])
        (ancestry, known, same), proposed, accepted, _ = labels.resolve(a, masks, np.full(3, 2))
        np.testing.assert_array_equal(ancestry[:, :3], [[1, 1, 1], [1, -1, 1], [-1, -1, -1]])
        np.testing.assert_array_equal(known[:, :2], [[6, 4], [0, 0], [0, 0]])
        np.testing.assert_array_equal(same[:, :2], [[2, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(ancestry[:, 3:], a["ancestryHiggs"][:, 3:])
        np.testing.assert_array_equal(proposed, accepted)
        np.testing.assert_array_equal(known[:, 2:], 0)

    def test_negative_tree_precedence_and_mixed_id_order(self):
        a = tree()
        a["ancestryHiggs"][0, :3] = [1, 1, 0]
        a["relationHiggs"][0, :3] = [2, 1, 0]
        a["higgsPairKnown"][0, :2] = [6, 4]
        (anc, known, same), _, _, _ = labels.resolve(a, np.array([[1, 1, 1]]), np.array([2]))
        np.testing.assert_array_equal(anc, a["ancestryHiggs"])
        np.testing.assert_array_equal(known, a["higgsPairKnown"])
        np.testing.assert_array_equal(same, a["higgsPairSame"])
        a = tree()
        a["relationHiggs"][0, :2] = [2, 1]
        a["ancestryHiggs"][0, :2] = 1
        (_, _, same), _, _, _ = labels.resolve(a, np.array([[1, 2, 1]]), np.array([2]))
        self.assertEqual(same[0, 0], 4)

    def test_unresolved_tie_with_definite_tree_pair(self):
        a = tree()
        a["higgsPairKnown"][0, 0] = 2
        out, p, keep, _ = labels.resolve(a, np.array([[1, 1, 1]]), np.array([2]))
        np.testing.assert_array_equal(p[0], [1, 1, 1, 0, 1, 1])
        np.testing.assert_array_equal(keep[0], [1, 1, 1, 0, 0, 0])
        self.assertFalse(labels.incompatible(np.array([2]), a["relationHiggs"], *out).any())
        # In a one-Higgs event the definite different pair rules out keeping both new ancestry tags.
        a = tree()
        a["higgsPairKnown"][0, 0] = 2
        _, _, keep, _ = labels.resolve(a, np.array([[1, 1, 0]]), np.array([1]))
        self.assertFalse(keep.any())

    def test_consistency_against_explicit_parent_assignments(self):
        gen = np.random.default_rng(42)
        slots, n = 4, 350
        count = gen.integers(1, 3, size=n)
        ids = gen.choice([-1, -1, -1, 1, 2], size=(n, slots))
        ids[(count == 1)[:, None] & (ids == 2)] = 1
        ancestry = gen.choice([-1, 0, 1], size=(n, slots))
        ancestry[ids > 0] = 1
        known = np.zeros_like(ids)
        same = known.copy()
        for i, j in itertools.combinations(range(slots), 2):
            has = gen.integers(0, 2, size=n)
            known[:, i] |= has << j
            same[:, i] |= (has & gen.integers(0, 2, size=n)) << j
        bad = labels.incompatible(count, ids, ancestry, known, same)
        for k in range(n):
            def feasible(x):
                if any((h > 0 and h != c) or (a >= 0 and a != int(c > 0))
                       for c, h, a in zip(x, ids[k], ancestry[k])):
                    return False
                return all(not (known[k,i] & (1 << j)) or
                           bool(same[k,i] & (1 << j)) == (x[i] > 0 and x[i] == x[j])
                           for i, j in itertools.combinations(range(slots), 2))
            assignments = itertools.product(range(int(count[k]) + 1), repeat=slots)
            self.assertEqual(bool(bad[k]), not any(feasible(x) for x in assignments))
