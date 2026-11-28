import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import yaml

from synapse.core.config import ModelConfig
from synapse.loss_functions.lepton_objects import EventAndLeptonObjectLoss, shared_parent_bce
from synapse.models.ParticleTransformer import ParticleTransformerSimple
from synapse.models.shared_parent import SharedParentHead


class SharedParentTests(unittest.TestCase):
    def test_explicit_higgs_pairs_train_with_ambiguous_ids_and_apply_vetoes(self):
        # The two events have no unique instance IDs. Only the first has real supervised pairs.
        loss = EventAndLeptonObjectLoss(origin_loss_weight=0, conversion_loss_weight=0,
            heavy_flavour_weight=0, balance_origin_classes=False, shared_parent_loss_weight=1,
            shared_parent_branches=["relationHiggs"],
            explicit_parent_pairs={"relationHiggs": ("known", "same")})
        shape = (2, 5)
        target = {key: torch.zeros(shape, dtype=torch.int16)
                  for key in ("known", "same", "isMET", "isLepton", "partOrigin", "isConv", "isHF")}
        target["relationHiggs"] = torch.full(shape, -1)
        target["isMET"][:, 3] = 1
        target["known"][:, 0] = torch.tensor([30, 24])
        target["same"][0, 0] = 2
        valid = torch.tensor([[[True, True, True, True, False]]]).repeat(2, 1, 1)
        weights = torch.tensor([1., 3.])
        loss.prepare_training_fold(target, valid, weights)
        self.assertEqual(float(loss.shared_parent_factors[0]), 4.)
        pairs = torch.zeros(2, 1, 5, 5, requires_grad=True)
        pred = (torch.zeros(2, 3), {"shared_parent_logits": pairs}, torch.zeros(10, 5))
        value, _ = loss(pred, torch.zeros(2, dtype=torch.long), weights, object_targets=target, object_mask=valid)
        value.backward()
        want = torch.zeros_like(pairs)
        want[0, 0, 0, 1] = -.25
        want[0, 0, 0, 2] = .25
        torch.testing.assert_close(pairs.grad, want)

    def test_earlier_recipe_and_checkpoint_restore_the_shared_parent_head(self):
        branches = {"relationZ": "p_pair_z", "relationHiggs": "p_pair_higgs"}
        recipe = {
            "model": "synapse.models.ParticleTransformer.ParticleTransformerSimple",
            "model_params": dict(input_dim=3, global_input_dim=2, num_classes=3, embed_dims=[8],
                                 global_embed_dims=[8], num_heads=2, num_layers=2, relation_rank=3,
                                 num_relations=2, relation_feedback={"met_feature_index": 1}),
            "relation_outputs": branches, "start_lr": 0.001,
            "loss_function": {"name": "synapse.loss_functions.lepton_objects.EventAndLeptonObjectLoss",
                              "params": {"relation_loss_weight": 0.02, "relation_branches": list(branches)}},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(yaml.safe_dump(recipe, sort_keys=False))
            config = ModelConfig(path)
        self.assertEqual(list(config.shared_parent_outputs.items()), list(branches.items()))
        loss = EventAndLeptonObjectLoss(**config.loss_function["params"])
        self.assertEqual(loss.shared_parent_branches, tuple(branches))
        self.assertEqual(loss.shared_parent_loss_weight, 0.02)

        # Lightning stores the model under a "model." prefix. Include nonzero trained feedback weights.
        model = torch.nn.ModuleDict({"model": ParticleTransformerSimple(**config.model_params)})
        with torch.no_grad():
            model["model"].shared_parent_head.feedback.weight.fill_(0.25)
        expected = model.state_dict()
        old_state = {key.replace("model.shared_parent_head.", "model.parent_head."): value
                     for key, value in expected.items()}
        restored = torch.nn.ModuleDict({"model": ParticleTransformerSimple(**config.model_params)})
        restored.load_state_dict(old_state, strict=True)
        torch.testing.assert_close(restored.state_dict(), expected, rtol=0, atol=0)

    def test_pair_scores_are_symmetric_and_follow_the_input_object_order(self):
        torch.manual_seed(2)
        head = SharedParentHead(embed_dim=8, rank=3, num_shared_parent_types=2)
        tokens = torch.randn(10, 12, 8)
        scores = head(tokens)
        self.assertEqual(scores.shape, (10, 2, 12, 12))
        torch.testing.assert_close(scores, scores.transpose(-1, -2))
        order = torch.randperm(12)
        torch.testing.assert_close(head(tokens[:, order]), scores[:, :, order][:, :, :, order])

    def test_parent_probabilities_enter_only_the_final_object_block(self):
        torch.manual_seed(7)
        model = ParticleTransformerSimple(
            input_dim=3, global_input_dim=2, num_classes=3, embed_dims=[8], global_embed_dims=[8],
            pair_embed_dims=[4], num_layers=5, num_cls_layers=1, num_heads=4, block_params={"num_heads": 2},
            shared_parent_rank=3, num_shared_parent_types=3, shared_parent_feedback={"veto_feature_indices": [1]}, fc_params=[],
        ).eval()
        # A normalized isMET column would identify the wrong objects with these statistics.
        model.setup_input_norm(torch.ones(3), torch.ones(3), torch.zeros(2), torch.ones(2))
        x, g = torch.randn(10, 3, 6), torch.randn(10, 2)
        x[:, 1] = 0
        x[::2, 1, 2] = 1
        x[1::2, 1, 4] = 1
        mask = torch.ones(10, 1, 6, dtype=torch.bool)
        mask[:, :, -1] = False
        momenta = torch.randn(10, 3, 6)
        v = torch.cat((momenta, momenta.norm(dim=1, keepdim=True) + 1), dim=1)
        target = torch.arange(10) % 3

        biases = []
        def record_bias(module, args, kwargs):
            biases.append(kwargs["attn_mask"])
        for block in model.blocks:
            block.register_forward_pre_hook(record_bias, with_kwargs=True)

        event, _, _ = model(x, g, v, mask)
        mixing = model.shared_parent_head.feedback.weight
        self.assertEqual(torch.count_nonzero(mixing).item(), 0)
        for bias in biases[1:]:
            torch.testing.assert_close(bias, biases[0])
        torch.nn.functional.cross_entropy(event, target).backward()
        self.assertGreater(mixing.grad.norm().item(), 0)
        self.assertEqual(model.shared_parent_head.projection.weight.grad.norm().item(), 0)

        # Give each head a distinct, signed mix of the same three parent channels.
        with torch.no_grad():
            mixing.copy_(torch.tensor([[1., -2., 0.], [0., 0.5, 1.]]))
        model.zero_grad(set_to_none=True)
        biases.clear()
        event, aux, _ = model(x, g, v, mask)
        shared_parent_bias = torch.einsum("hr,brij->bhij", mixing, aux["shared_parent_logits"].sigmoid())
        eligible = mask[:, 0] & (x[:, 1] == 0)
        pairs = eligible[:, :, None] & eligible[:, None, :] & ~torch.eye(6, dtype=torch.bool)
        expected = (shared_parent_bias * pairs[:, None]).flatten(0, 1)
        for bias in biases[1:-1]:
            torch.testing.assert_close(bias, biases[0])
        torch.testing.assert_close(biases[-1], biases[0] + expected)
        torch.nn.functional.cross_entropy(event, target).backward()
        self.assertGreater(model.shared_parent_head.projection.weight.grad.norm().item(), 0)

        # An empty veto list keeps the learned feedback, including for MET. Padding and self-pairs stay masked.
        model.shared_parent_veto_indices = ()
        biases.clear()
        with torch.no_grad():
            model(x, g, v, mask)
        valid = mask[:, 0]
        pairs = valid[:, :, None] & valid[:, None, :] & ~torch.eye(6, dtype=torch.bool)
        torch.testing.assert_close(biases[-1], biases[0] + (shared_parent_bias * pairs[:, None]).flatten(0, 1))

    def test_veto_features_follow_column_names_and_control_pair_supervision(self):
        recipe = Path(__file__).resolve().parents[1] / "configs/v06_test_run3/model_config.yaml"
        config = ModelConfig(recipe)
        data = SimpleNamespace(inputs={"obj_feats": ["isJet", "pt", "isMET"]})
        valid = torch.tensor([[True, True, True, False]]).expand(2, -1)
        targets = {"relationHiggs": torch.tensor([[1, 1, -1, 1], [1, -1, 1, 1]]),
                   "isMET": torch.tensor([[0, 1, 0, 0]]).expand(2, -1),
                   "isJet": torch.tensor([[0, 0, 1, 0]]).expand(2, -1)}
        weights = torch.tensor([1., 3.])
        # The only known pair in event 0 uses MET. The only one in event 1 uses the jet.
        cases = [(["isMET"], [2], [False, True], 4/3),
                 (["isJet"], [0], [True, False], 4.),
                 (["isMET", "isJet"], [2, 0], [False, False], 0.),
                 ([], [], [True, True], 1.)]
        for flags, indices, has_pair, factor in cases:
            with self.subTest(flags=flags):
                config.shared_parent_veto_features = flags
                feedback = config.model_params_for(data)["shared_parent_feedback"]
                self.assertEqual(feedback["veto_feature_indices"], indices)
                self.assertEqual(config.loss_function["params"]["shared_parent_veto_features"], flags)
                loss = EventAndLeptonObjectLoss(balance_origin_classes=False, shared_parent_loss_weight=0.02,
                                                shared_parent_branches=["relationHiggs"], shared_parent_veto_features=flags)
                loss.prepare_training_fold(targets, valid[:, None], weights)
                self.assertAlmostEqual(loss.shared_parent_factors.item(), factor, places=6)
                ids = loss._parent_ids(targets, valid)
                costs = shared_parent_bce(torch.zeros(2, 1, 4, 4), ids)
                self.assertEqual((costs[:, 0] > 0).tolist(), has_pair)
        config.shared_parent_veto_features = ["missingFlag"]
        with self.assertRaisesRegex(ValueError, "missingFlag"):
            config.model_params_for(data)

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
                                           balance_origin_classes=False, shared_parent_loss_weight=0.02,
                                           shared_parent_branches=branches)
        loss_fn.prepare_training_fold(targets, valid, weights)
        torch.testing.assert_close(loss_fn.shared_parent_factors, torch.tensor([0., 7/2, 7/4, 7/3]))
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
