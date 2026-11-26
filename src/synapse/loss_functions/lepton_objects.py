import torch
from torch.nn import functional as F

from .loss import CrossEntropyLossWeighted


def _fold_chunks(n_events):
    # At 10 slots, a float64 [events, slots, 3 origins] product takes 480 MB for 2M events.
    # Each chunk needs about 16 MB for that product. The sums accumulate over the whole fold.
    chunk_size = 65_536
    for start in range(0, n_events, chunk_size):
        yield slice(start, start + chunk_size)


class EventAndLeptonObjectLoss(CrossEntropyLossWeighted):
    """Event truth CE plus binary origin, conversion and heavy-flavour losses during training."""

    def __init__(self, origin_loss_weight=0.10, conversion_loss_weight=0.10,
                 heavy_flavour_weight=0.08, balance_origin_classes=True):
        """Set how much the origin, conversion and heavy-flavour losses contribute to training."""
        super().__init__()
        if min(origin_loss_weight, conversion_loss_weight, heavy_flavour_weight) < 0:
            raise ValueError("Object loss weights must be nonnegative")
        self.origin_loss_weight = origin_loss_weight
        self.conversion_loss_weight = conversion_loss_weight
        self.heavy_flavour_weight = heavy_flavour_weight
        # Rows hold negative and positive weights. Columns correspond to Higgs, top and W/Z/VV.
        self.origin_class_weights = torch.ones(2, 3) if balance_origin_classes else None
        self.mean_valid_slots = None  # Fitted on the training fold, including when resuming.

    @staticmethod
    def origin_targets(origin, lepton):
        """Convert MC origin codes of shape [events, slots] to three binary targets per slot.

        The output order is Higgs (code 1), top (2), then W/Z/diboson (3 through 5).
        Only slots selected by the lepton mask can have a positive target.
        """
        return torch.stack((lepton & (origin == 1), lepton & (origin == 2),
                            lepton & (origin >= 3) & (origin <= 5)), dim=-1)

    @torch.no_grad()
    def prepare_training_fold(self, targets, mask, event_weights):
        """Fit the object-loss normalization and optional origin class weights in one pass.

        All valid objects enter the normalization. Each origin head balances its positive
        and negative reconstructed leptons using sums of training event weights (SOW).
        """
        event_weights = torch.as_tensor(event_weights, dtype=torch.float64)
        balance_origins = self.origin_class_weights is not None
        event_sow = 0.0
        object_sow = 0.0
        positive_sow = event_weights.new_zeros(3)
        negative_sow = torch.zeros_like(positive_sow)

        for rows in _fold_chunks(len(event_weights)):
            weights = event_weights[rows]
            if not torch.all(torch.isfinite(weights) & (weights >= 0)):
                raise ValueError("Object training requires finite, nonnegative event weights")
            valid = torch.as_tensor(mask[rows]).squeeze(1).bool()
            # An event of weight w with n valid slots adds w to event_sow and w * n to object_sow.
            # Each valid lepton, jet and MET slot inherits w. Padding gets zero.
            object_weights = weights[:, None] * valid
            event_sow += weights.sum()
            object_sow += object_weights.sum()

            if balance_origins:
                # isLepton selects reconstructed electrons and muons. partOrigin supplies their MC label.
                is_lepton = torch.as_tensor(targets["isLepton"][rows]).bool()
                origin_codes = torch.as_tensor(targets["partOrigin"][rows])
                positive = self.origin_targets(origin_codes, is_lepton)  # [events, slots, 3 origins]
                lepton_weights = (object_weights * is_lepton).unsqueeze(-1)  # [events, slots, 1]
                # Summing over events and slots leaves one total per origin. For the Higgs head,
                # for example, all other reconstructed leptons are negatives. Jets and MET add zero here.
                positive_sow += (lepton_weights * positive).sum(dim=(0, 1))
                negative_sow += (lepton_weights * ~positive).sum(dim=(0, 1))

        if balance_origins:
            if not torch.all((positive_sow > 0) & (negative_sow > 0)):
                raise ValueError("Origin balancing needs positive and negative lepton weight for Higgs, top and W/Z/VV")
            # Fit once from the whole fold. Each group gets half its origin's combined lepton weight.
            # For example, positive and negative sums of 2 and 8 both become 5 with factors 5/2 and 5/8.
            target_sow = (positive_sow + negative_sow) / 2
            negative_factor = target_sow / negative_sow
            positive_factor = target_sow / positive_sow
            self.origin_class_weights = torch.stack((negative_factor, positive_factor)).float()

        # sum(w * n_valid) / sum(w) is the weighted mean number of valid objects per event.
        # Every training batch uses this divisor, including when class balancing is disabled.
        self.mean_valid_slots = float(object_sow / event_sow) if object_sow > 0 else 1.0


    def forward(self, pred, target, weight, mean_train_weight=None, object_targets=None, object_mask=None, **kwargs):
        """Return the batch loss and the event-class probabilities computed from pred[0].

        Training adds the object losses to the weighted event cross-entropy. Validation and
        testing use the event cross-entropy alone, so checkpoint selection measures the event task.
        """
        event_loss, scores = super().forward(pred, target, weight, mean_train_weight)
        if not self.training:
            return event_loss, scores
        if object_targets is None or object_mask is None:
            raise ValueError("Object training requires named per-object truth and a validity mask")
        valid = object_mask.squeeze(1).bool()
        if pred[2].shape != (valid.numel(), 5):
            raise ValueError("Expected five object logits per input slot")
        lepton = valid & object_targets["isLepton"].bool()
        origins = self.origin_targets(object_targets["partOrigin"], lepton)
        conv = lepton & (object_targets["isConv"] == 1)
        hf = lepton & (object_targets["isHF"] == 1)
        # Align pred[2] from [events * slots, 5] with truth of shape [events, slots, 5].
        # Each logit gets its own binary cross-entropy in the order Higgs, top, W/Z/VV, conversion, HF.
        truth = torch.cat((origins, conv.unsqueeze(-1), hf.unsqueeze(-1)), dim=-1)
        logits = pred[2].reshape_as(truth)
        bce = F.binary_cross_entropy_with_logits(logits, truth.to(logits.dtype), reduction="none")

        origin_bce = bce[:, :, :3]
        if self.origin_class_weights is not None:
            self.origin_class_weights = self.origin_class_weights.to(logits.device)
            negative, positive = self.origin_class_weights.unbind()
            lepton_factors = torch.where(origins, positive, negative)
            # Jets and MET have zero origin targets and keep their event weight (class factor 1).
            origin_bce = origin_bce * lepton_factors.masked_fill(~lepton.unsqueeze(-1), 1.0)
        # Average the three origin heads per object, then sum the valid objects in each event.
        origin_sum = origin_bce.mean(dim=-1).masked_fill(~valid, 0).sum(dim=1)
        mean_valid_slots = self.mean_valid_slots
        if mean_valid_slots is None:
            # Standalone calls without a fitted fold use one shared divisor for the batch.
            object_sow = (weight * valid.sum(dim=1)).sum()
            mean_valid_slots = torch.where(object_sow > 0, object_sow / weight.sum(), 1.0)
        # A shared divisor keeps the fold-average loss on the scale of one object's BCE.
        # Leptons with the same event weight and origin label keep the same coefficient at any jet count.
        origin_loss = origin_sum / mean_valid_slots
        # Conversion and heavy flavour have truth labels only for leptons, so use the lepton count here.
        lepton_bce = bce[:, :, 3:].masked_fill(~lepton.unsqueeze(-1), 0)
        conv_loss, hf_loss = (lepton_bce.sum(dim=1) / lepton.sum(dim=1, keepdim=True).clamp_min(1)).unbind(dim=1)
        auxiliary = (self.origin_loss_weight * origin_loss + self.conversion_loss_weight * conv_loss
                     + self.heavy_flavour_weight * hf_loss)
        # Apply the same event weights and normalization as the event loss. The recipe supplies the training-fold mean.
        normalizer = weight.mean() if mean_train_weight is None else mean_train_weight
        return event_loss + (weight * auxiliary).mean() / normalizer, scores

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints stored these factors. Training refits them on the current fold at startup.
        state_dict.pop(prefix + "origin_class_weights", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
