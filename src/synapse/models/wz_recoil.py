"""Independent transverse WZ recoil and control-region gradient isolation."""
from contextlib import contextmanager
import torch
from torch import nn

class WZRecoil(nn.Module):
    """Evaluate each SFOS hypothesis independently using only transverse recoil observables."""

    def __init__(self):
        super().__init__()
        self.rank = nn.Sequential(nn.Linear(13, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, z, remaining, met, jets, jet_valid, sfos):
        # All momentum inputs are transverse (px, py) in GeV. The individual Z daughters never enter.
        def pt(v):
            return torch.linalg.vector_norm(v, dim=-1)

        def cos(a, b):
            return ((a * b).sum(-1) / (pt(a) * pt(b)).clamp_min(1.0e-6)).clamp(-1, 1)

        jets = jets * jet_valid[..., None]
        recoil = jets.sum(1)[:, None]
        ht = pt(jets).sum(1, keepdim=True)
        count = jet_valid.sum(1, keepdim=True)
        w = remaining + met
        mt = (2 * pt(remaining) * pt(met) * (1 - cos(remaining, met))).clamp_min(0).sqrt()
        zpt = pt(z)
        feats = [
            zpt.log1p(),
            pt(w).log1p(),
            pt(remaining).log1p(),
            pt(met).expand_as(zpt).log1p(),
            mt.log1p(),
            cos(z, w),
            cos(remaining, met),
            cos(z, recoil),
            cos(w, recoil),
            ht.expand_as(zpt).log1p(),
            count.expand_as(zpt),
            pt(z + w).log1p(),
            pt(z + w + recoil).log1p(),
        ]
        candidates = self.rank(torch.stack(feats, -1)).squeeze(-1)
        # Uniform hypotheses for WZ and background. No mass-based or truth-based candidate selection.
        return (candidates * sfos).sum(1) / sfos.sum(1).clamp_min(1)

    def score_event(self, features, vectors, mask):
        transverse = vectors[:, :2].transpose(1, 2).float() * 0.001
        valid = mask[:, 0].bool()
        i, j, remaining = [0, 0, 1], [1, 2, 2], [2, 1, 0]
        pdg = features[:, 4]
        sfos = (pdg[:, i] == -pdg[:, j]) & valid[:, i] & valid[:, j]
        jets = valid & features[:, 7].bool()
        return self(transverse[:, i]+transverse[:, j], transverse[:, remaining],
                    transverse[:, 9:10], transverse, jets, sfos)


@contextmanager
def frozen_batch_norm(model):
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
    try:
        for layer in layers:
            layer.eval()
        yield
    finally:
        for layer in layers:
            layer.train()


def forward_with_control(model, inputs, control):
    base, cr = (~control).nonzero().flatten(), control.nonzero().flatten()
    if not len(cr) and len(base)>1:
        return model(*inputs)
    groups = []
    if len(base)>1:
        groups.append(model(*(x[base] for x in inputs)))
    with frozen_batch_norm(model):
        if len(base)==1:
            groups.append(model(*(x[base] for x in inputs)))
        if len(cr):
            groups.append(model(*(x[cr] for x in inputs), detach_relation_inputs=True))
    if len(groups)==1:
        return groups[0]
    order = torch.cat((base,cr)).argsort()
    scores, aux, objects = zip(*groups)
    slots = inputs[0].size(-1)
    obj = torch.cat([o.reshape(-1,slots,model.num_obj_classes) for o in objects])[order]
    return torch.cat(scores)[order], {k: torch.cat([g[k] for g in aux])[order] for k in aux[0]}, obj.flatten(0,1)
