import math

import torch
from torch import nn


# Earlier recipes excluded MET from the four shared-parent matrices.
DEFAULT_SHARED_PARENT_VETO = ("isMET",)


def shared_parent_object_mask(valid, veto_flags):
    """Select valid objects for which every configured veto flag is false."""
    eligible = valid.bool()
    for flag in veto_flags:
        eligible = eligible & ~torch.as_tensor(flag, device=valid.device).bool()
    return eligible


class SharedParentHead(nn.Module):
    """Predict one symmetric object-pair matrix per configured parent type."""

    def __init__(self, embed_dim, rank, num_shared_parent_types, feedback_heads=None):
        super().__init__()
        if rank < 1 or num_shared_parent_types < 1:
            raise ValueError("The shared-parent head needs a positive rank and at least one parent type")
        self.rank = rank
        self.num_shared_parent_types = num_shared_parent_types
        self.projection = nn.Linear(embed_dim, 2 * num_shared_parent_types * rank, bias=False)
        self.intercept = nn.Parameter(torch.zeros(num_shared_parent_types))
        self.feedback = None
        if feedback_heads is not None:
            self.feedback = nn.Linear(num_shared_parent_types, feedback_heads, bias=False)
            nn.init.zeros_(self.feedback.weight)

    def forward(self, tokens):
        # Each object gets a left and right vector of length 'rank' for each parent type.
        batch_size, max_objects, _ = tokens.shape  # max_objects includes padding.
        vectors = self.projection(tokens).reshape(batch_size, max_objects, 2, self.num_shared_parent_types, self.rank)
        left, right = vectors.unbind(dim=2)  # Both [batch_size, max_objects, num_shared_parent_types, rank].
        left, right = left.transpose(1, 2), right.transpose(1, 2)
        # Score (i, j) in both orders and average. Each [batch, parent] matrix is symmetric.
        scores = left @ right.transpose(-1, -2)
        symmetric = (scores + scores.transpose(-1, -2)) / (2 * math.sqrt(self.rank))
        return symmetric + self.intercept[None, :, None, None]  # [batch_size, num_shared_parent_types, max_objects, max_objects]

    def attention_bias(self, logits, eligible_objects):
        """Mix the parent probabilities into one additive attention bias per head and object pair."""
        valid_pairs = eligible_objects[:, :, None] & eligible_objects[:, None, :]
        self_pairs = torch.eye(logits.size(-1), dtype=torch.bool, device=logits.device)
        probabilities = logits.sigmoid().masked_fill(~valid_pairs[:, None] | self_pairs, 0)
        # Linear mixes only the parent axis: [batch, i, j, parent] -> [batch, i, j, head].
        bias = self.feedback(probabilities.movedim(1, -1))
        return bias.movedim(-1, 1).flatten(0, 1)  # MultiheadAttention expects [batch * heads, i, j].
