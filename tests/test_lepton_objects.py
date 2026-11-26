import math
import unittest
from unittest.mock import patch

import torch

from synapse.loss_functions.lepton_objects import EventAndLeptonObjectLoss
from synapse.models.ParticleTransformer import LeptonJetHeavyFlavourHead, ParticleTransformerSimple


class LeptonObjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_ten_event_training_step(self):
        torch.manual_seed(10)
        model = ParticleTransformerSimple(
            input_dim=10, global_input_dim=2, num_classes=3, num_obj_classes=5,
            lepton_jet_hf={"jet_tag_indices": [8, 9], "use_event_context": True},
            embed_dims=[8], global_embed_dims=[8],
            pair_embed_dims=[4], num_heads=2, num_layers=1, num_cls_layers=1, fc_params=[[8, 0]],
        )
        model.setup_input_norm(torch.zeros(10), torch.ones(10), torch.zeros(2), torch.ones(2))
        events, slots = 10, 10
        features = torch.randn(events, 10, slots)
        vectors = torch.randn(events, 4, slots) * 1000
        vectors[:, 3] = vectors[:, :3].norm(dim=1) + 100
        mask = torch.ones(events, 1, slots, dtype=torch.bool)
        origins = torch.zeros(events, slots, dtype=torch.long)
        origins[:, :3] = torch.tensor([1, 2, 3])
        is_lepton = torch.zeros_like(origins, dtype=torch.bool)
        is_lepton[:, :3] = True
        targets = {"partOrigin": origins, "isLepton": is_lepton,
                   "isConv": origins == 1, "isHF": origins == 2}
        weights = torch.ones(events)
        loss_fn = EventAndLeptonObjectLoss()
        loss_fn.prepare_training_fold(targets, mask, weights)
        pred = model(features, torch.randn(events, 2), vectors, mask)
        event_targets = torch.arange(events) % 3
        # Isolate the event loss so auxiliary supervision cannot account for these gradients.
        event_loss = torch.nn.functional.cross_entropy(pred[0], event_targets)
        event_layers = (model.hf_event_projection, model.hf_head.pair_rank, model.hf_head.classifier[-1])
        grads = torch.autograd.grad(event_loss, [layer.weight for layer in event_layers], retain_graph=True)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue(grad.ne(0).any())

        loss, _ = loss_fn(pred, event_targets, weights, 1., object_targets=targets, object_mask=mask)
        loss.backward()
        for layer in (model.obj_fc[-1], model.hf_head.classifier[-1], model.hf_head.pair_rank):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertTrue(layer.weight.grad.ne(0).any(dim=1).all())

    def test_binary_targets_weights_and_masks(self):
        loss_fn = EventAndLeptonObjectLoss()
        mask = torch.tensor([[[1, 1, 1, 1, 0]]], dtype=torch.bool)
        targets = {
            "partOrigin": torch.tensor([[1, 2, 4, 1, -1]]),
            "isLepton": torch.tensor([[1, 1, 1, 0, 0]]),
            "isConv": torch.tensor([[0, 1, 0, 1, 1]]),
            "isHF": torch.tensor([[1, 0, 0, 1, 1]]),
        }
        weight = torch.tensor([2.])
        loss_fn.prepare_training_fold(targets, mask, weight)
        torch.testing.assert_close(loss_fn.origin_class_weights,
                                   torch.tensor([[0.75, 0.75, 0.75], [1.5, 1.5, 1.5]]))
        logits = torch.zeros(5, 5, requires_grad=True)
        pred = (torch.zeros(1, 3), None, logits)
        truth = torch.tensor([0])
        loss, _ = loss_fn(pred, truth, weight, 4., object_targets=targets, object_mask=mask)
        self.assertAlmostEqual(loss.item(), (math.log(3) + 0.28 * math.log(2)) / 2, places=6)
        loss.backward()
        self.assertEqual(logits.grad.sign().tolist(), [
            [-1, 1, 1, 1, -1], [1, -1, 1, -1, 1], [1, 1, -1, 1, 1],
            [1, 1, 1, 0, 0], [0, 0, 0, 0, 0],
        ])
        event_loss, _ = loss_fn.eval()(pred, truth, weight, 4.)
        self.assertAlmostEqual(event_loss.item(), math.log(3) / 2, places=6)

    def test_origin_object_weight_is_independent_of_jet_count(self):
        weights = torch.tensor([1., 1., 2.])
        valid_counts = torch.tensor([4, 8, 4])
        mask = (torch.arange(8) < valid_counts[:, None]).unsqueeze(1)
        origins = torch.zeros(3, 8, dtype=torch.long)
        origins[:, :3] = torch.tensor([[1, 2, 0], [1, 0, 3], [1, 2, 0]])
        is_lepton = torch.zeros_like(origins, dtype=torch.bool)
        is_lepton[:, :3] = True  # A reconstructed lepton with code 0 is negative for all three origins.
        targets = {"partOrigin": origins, "isLepton": is_lepton,
                   "isConv": torch.zeros_like(origins), "isHF": torch.zeros_like(origins)}

        for balance in (True, False):
            with self.subTest(balance=balance):
                loss_fn = EventAndLeptonObjectLoss(origin_loss_weight=1., balance_origin_classes=balance)
                # Two uneven chunks exercise accumulation across boundaries in a single fold scan.
                with patch("synapse.loss_functions.lepton_objects._fold_chunks",
                           return_value=(slice(0, 2), slice(2, 3))) as chunks:
                    loss_fn.prepare_training_fold(targets if balance else {}, mask, weights)
                    chunks.assert_called_once_with(len(weights))
                self.assertEqual(loss_fn.mean_valid_slots, 5.)  # 20 weighted objects / 4 event weight.
                if balance:
                    # The positive sums are 4, 3 and 1 out of a total lepton weight of 12.
                    torch.testing.assert_close(loss_fn.origin_class_weights, torch.tensor([
                        [6/8, 6/9, 6/11], [6/4, 6/3, 6.],
                    ]))

                # Put the identical Higgs logits in separate batches to exercise the fold normalization.
                logits = torch.zeros(3, 8, 5, requires_grad=True)
                for i in range(3):
                    event = slice(i, i + 1)
                    event_targets = {key: values[event] for key, values in targets.items()}
                    pred = (torch.zeros(1, 3), None, logits[i])
                    loss, _ = loss_fn(pred, torch.tensor([0]), weights[event], weights.mean(),
                                      object_targets=event_targets, object_mask=mask[event])
                    loss.backward()
                unit_grad = 0.5 * weights / (3 * 5 * weights.mean())
                higgs_factor = 6/4 if balance else 1.
                torch.testing.assert_close(logits.grad[:, 0, 0], -higgs_factor * unit_grad)
                # Slot 3 is a valid non-lepton with unit class weight for every origin.
                torch.testing.assert_close(logits.grad[:, 3, :3], unit_grad[:, None].expand(-1, 3))

    def test_heavy_flavour_excludes_padded_jets_met_and_forward_slots(self):
        torch.manual_seed(7)
        head = LeptonJetHeavyFlavourHead(8, [8, 9])
        tokens = torch.randn(3, 12, 8, requires_grad=True)
        features = torch.randn(3, 10, 12)
        vectors = torch.randn(3, 4, 12) * 1000
        valid = torch.ones(3, 12, dtype=torch.bool)
        valid[0, 3:9] = False  # No central jet for this event.
        valid[1, 2] = False    # One padded lepton, two valid central jets.
        valid[1, 5:9] = False
        valid[2, :3] = False   # Jets present, no valid leptons.
        logits, event_jet_context = head(tokens, features, vectors, valid)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits[:, 3:] == -20).all())
        self.assertTrue((logits[~valid] == -20).all())
        self.assertEqual(event_jet_context.shape, (3, 48))
        self.assertTrue(torch.isfinite(event_jet_context).all())
        self.assertTrue((event_jet_context[[0, 2]] == 0).all())
        (logits[:, :3].sum() + event_jet_context.sum()).backward()
        self.assertEqual(tokens.grad[0, 3:].count_nonzero().item(), 0)
        self.assertEqual(tokens.grad[1, 2].count_nonzero().item(), 0)
        self.assertEqual(tokens.grad[1, 5:].count_nonzero().item(), 0)
        self.assertGreater(tokens.grad[1, 3:5].abs().sum().item(), 0)
        self.assertEqual(tokens.grad[2].count_nonzero().item(), 0)
