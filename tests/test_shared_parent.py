import math
import unittest

import torch

from synapse.loss_functions.lepton_objects import EventAndLeptonObjectLoss
from synapse.models.shared_parent import SharedParentHead


class SharedParentTests(unittest.TestCase):
    def test_pair_scores_are_symmetric_and_follow_the_input_object_order(self):
        torch.manual_seed(2)
        head = SharedParentHead(embed_dim=8, rank=3, num_relations=2)
        tokens = torch.randn(10, 12, 8)
        scores = head(tokens)
        self.assertEqual(scores.shape, (10, 2, 12, 12))
        torch.testing.assert_close(scores, scores.transpose(-1, -2))
        order = torch.randperm(12)
        torch.testing.assert_close(head(tokens[:, order]), scores[:, :, order][:, :, :, order])

    def test_masked_pairs_and_per_parent_fold_weights(self):
        events, max_objects = 3, 7
        known = torch.tensor([1, 1, 2, 1, -1, 1, 0], dtype=torch.int8)
        empty = torch.full_like(known, -1)
        two_nonmembers = torch.tensor([0, 0, -1, -1, -1, -1, -1], dtype=torch.int8)
        weights = torch.tensor([2., 1., 4.])
        valid = torch.ones(events, 1, max_objects, dtype=torch.bool)
        valid[:, :, 5] = False
        targets = {key: torch.zeros(events, max_objects, dtype=torch.int8)
                   for key in ("partOrigin", "isLepton", "isConv", "isHF", "isMET")}
        targets["isMET"][:, 3] = 1
        targets.update(relationHiggs=torch.stack((known, two_nonmembers, empty)),
                       relationTop=torch.stack((known, empty, empty)),
                       relationW=torch.stack((empty, empty, known)),
                       relationZ=torch.stack((empty, empty, empty)))

        # Select channels in a different order from the input dictionary.
        branches = ["relationZ", "relationTop", "relationW", "relationHiggs"]
        loss_fn = EventAndLeptonObjectLoss(origin_loss_weight=0, conversion_loss_weight=0, heavy_flavour_weight=0,
                                           balance_origin_classes=False, relation_loss_weight=0.02,
                                           relation_branches=branches)
        loss_fn.prepare_training_fold(targets, valid, weights)
        torch.testing.assert_close(loss_fn.relation_factors, torch.tensor([0., 7/2, 7/4, 7/3]))
        logits = torch.zeros(events, len(branches), max_objects, max_objects, requires_grad=True)
        pred = (torch.zeros(events, 3), {"shared_parent_logits": logits}, torch.zeros(events * max_objects, 5))
        loss, _ = loss_fn(pred, torch.zeros(events, dtype=torch.long), weights, weights.mean(),
                          object_targets=targets, object_mask=valid)
        # With zero logits each of the three labelled parent channels contributes 0.02 * log(2).
        self.assertAlmostEqual(loss.item(), math.log(3) + 3 * 0.02 * math.log(2), places=6)
        loss.backward()
        higgs = logits.grad[0, 3]
        self.assertLess(higgs[0, 1].item(), 0)       # Same positive ID.
        self.assertGreater(higgs[0, 2].item(), 0)    # Different positive IDs.
        self.assertGreater(higgs[0, 6].item(), 0)    # ID 1 paired with known nonmember 0.
        self.assertGreater(logits.grad[1, 3, 0, 1].item(), 0)  # Two zero IDs are also a negative pair.
        self.assertFalse(higgs[:, [3, 4, 5]].any())  # MET, unknown, padding.
        self.assertFalse(higgs[[3, 4, 5], :].any())
        self.assertFalse(logits.grad.tril().any())   # Diagonal and duplicate order.
        self.assertFalse(logits.grad[:, 0].any())    # No Z labels in this fold.
