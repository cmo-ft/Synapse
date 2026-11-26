import math

import torch
from torch import nn


class SharedParentHead(nn.Module):
    """Predict one symmetric object-pair matrix per configured parent type."""

    def __init__(self, embed_dim, rank, num_relations):
        super().__init__()
        if rank < 1 or num_relations < 1:
            raise ValueError("The shared-parent head needs a positive rank and at least one relation")
        self.rank = rank
        self.num_relations = num_relations
        self.projection = nn.Linear(embed_dim, 2 * num_relations * rank, bias=False)
        self.intercept = nn.Parameter(torch.zeros(num_relations))

    def forward(self, tokens):
        # Each object gets a left and right vector of length 'rank' for each parent type.
        batch_size, max_objects, _ = tokens.shape  # max_objects includes padding.
        vectors = self.projection(tokens).reshape(batch_size, max_objects, 2, self.num_relations, self.rank)
        left, right = vectors.unbind(dim=2)  # Both [batch_size, max_objects, num_relations, rank].
        left, right = left.transpose(1, 2), right.transpose(1, 2)
        # Score (i, j) in both orders and average. Each [batch, parent] matrix is symmetric.
        scores = left @ right.transpose(-1, -2)
        symmetric = (scores + scores.transpose(-1, -2)) / (2 * math.sqrt(self.rank))
        return symmetric + self.intercept[None, :, None, None]  # [batch_size, num_relations, max_objects, max_objects]
