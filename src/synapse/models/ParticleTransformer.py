""" Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial

from .shared_parent import SharedParentHead, shared_parent_object_mask


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    # rapidity = 0.5 * torch.log((energy + pz) / (energy - pz))
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)


def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = min(1, random.uniform(*self.target))
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        # x: (seq_len, batch, embed_dim)
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.no_grad():
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)
            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.head_scale = nn.Parameter(torch.ones(num_heads)) if scale_heads else None
        self.register_parameter("c_attn", None)  # Checkpoints with post-projection head scaling.
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The parameter name records which operation was trained, including after re-saving.
        if self.head_scale is not None and prefix + "c_attn" in state_dict:
            self.c_attn, self.head_scale = self.head_scale, None
        elif self.c_attn is not None and prefix + "head_scale" in state_dict:
            self.head_scale, self.c_attn = self.c_attn, None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _attention(self, query, key_value, padding_mask, attn_mask=None):
        attn = self.attn
        if self.head_scale is None:
            x = attn(query, key_value, key_value, key_padding_mask=padding_mask, attn_mask=attn_mask)[0]
            if self.c_attn is not None:
                heads = x.reshape(*x.shape[:2], self.num_heads, self.head_dim)
                x = torch.einsum('tbhd,h->tbdh', heads, self.c_attn).reshape_as(x)
            return x

        # Scaling the columns of W_out is equivalent to scaling each head before mixing heads.
        projection = attn.out_proj.weight
        columns = projection.reshape(self.embed_dim, self.num_heads, self.head_dim)
        scaled_projection = (columns * self.head_scale[None, :, None]).reshape_as(projection)
        output, _ = nn.functional.multi_head_attention_forward(
            query, key_value, key_value, self.embed_dim, self.num_heads,
            attn.in_proj_weight, attn.in_proj_bias, attn.bias_k, attn.bias_v,
            attn.add_zero_attn, attn.dropout, scaled_projection, attn.out_proj.bias,
            training=attn.training, key_padding_mask=padding_mask, attn_mask=attn_mask,
        )
        return output

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            if padding_mask is not None:
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self._attention(x_cls, u, padding_mask)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self._attention(x, x, padding_mask, attn_mask)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output

class GlobalEmbed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.Linear(input_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim)
            x = self.input_bn(x)
        x = self.embed(x)

        return x

class BlockGlobalFeat(Block):
    def forward(self, x, x_cls=None, x_glob=None, padding_mask=None, attn_mask=None):
        if x_glob is None:
            x_cls = None
        elif x_cls is not None:
            # Block prepends the class token. The global token is its first key/value.
            x = torch.cat((x_glob, x), dim=0)
            if padding_mask is not None:
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
        return super().forward(x, x_cls, padding_mask, attn_mask)


class _NormalizedTransformer(nn.Module):
    """Store training-fold statistics with the model and restore legacy checkpoints without them."""

    def __init__(self, input_dim, global_input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.global_input_dim = global_input_dim
        obj_shape, evt_shape = (1, input_dim, 1), (1, global_input_dim)
        self._norm_shapes = dict(x_mean=obj_shape, x_std=obj_shape, g_mean=evt_shape, g_std=evt_shape)
        for name in self._norm_shapes:
            self.register_buffer(name, None)

    @property
    def has_input_norm(self):
        return self.x_mean is not None

    def setup_input_norm(self, x_mean, x_std, g_mean, g_std):
        stats = dict(x_mean=x_mean, x_std=x_std, g_mean=g_mean, g_std=g_std)
        buffers = {name: stats[name].detach().reshape(shape).to(self.cls_token)
                   for name, shape in self._norm_shapes.items()}
        for name, value in buffers.items():
            setattr(self, name, value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        present = [prefix + name in state_dict for name in self._norm_shapes]
        complete = all(present)
        if any(present) and not complete:
            error_msgs.append(f"Incomplete input normalization in checkpoint at {prefix!r}")
        # PyTorch skips None buffers. Allocate the expected shapes only for stored statistics.
        for name, shape in self._norm_shapes.items():
            setattr(self, name, self.cls_token.new_empty(shape) if complete else None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def normalize_inputs(self, x, g):
        if not self.has_input_norm:
            raise RuntimeError("Call setup_input_norm with the training-fold statistics before scoring.")
        return (x - self.x_mean) / self.x_std, (g - self.g_mean) / self.g_std


class ParticleTransformerGlobalFeat(_NormalizedTransformer):

    def __init__(self,
                 input_dim,
                 global_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 global_embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[[64,0], [256,0], [64,0]],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(input_dim, global_input_dim, **kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        assert global_embed_dims[-1] == embed_dims[-1] if len(global_embed_dims) > 0 else input_dim, \
            'The last dimension of global_embed_dims must match the last dimension of embed_dims.'
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.global_embed = GlobalEmbed(global_input_dim, global_embed_dims, activation=activation) if len(global_embed_dims) > 0 else nn.Identity()
        self.blocks = nn.ModuleList([BlockGlobalFeat(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([BlockGlobalFeat(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = 2*embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # fc for object classification 
        # self.num_obj_classes = 4 # hardcoded for now:  0 for others, 1 for higgs, 2 for top, 3 for z, w, diboson
        self.num_obj_classes = 2 # hardcoded for now:  0 for others, 1 for heavy flavours
        self.obj_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, self.num_obj_classes)
        )
        self.fc_for_aux = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 2)
        )
        self.fc_for_aux2 = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 2)
        )

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, g, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # g: (N, C) -- global feature
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        # normalize input features
        x, g = self.normalize_inputs(x, g)

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            g = self.global_embed(g)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, x_glob=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            g = g.expand(1, x.size(1), -1) # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, x_glob=g, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0) # (N, C)
            g = g.squeeze(0)
            x_cls = torch.cat((x_cls, g), dim=1)
            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            # object output: classification for each particle
            obj_output = self.obj_fc(x.permute(1,0,2).contiguous().view(-1, x.size(2))) # (N*P, num_obj_classes)
            # if self.for_inference:
            #     output = torch.softmax(output, dim=1)
            #     return output
            # print('output:\n', output)
            output_for_aux = self.fc_for_aux(x_cls)
            output_for_aux2 = self.fc_for_aux2(x_cls)
            # output_for_aux = torch.cat((output_for_aux, output_for_aux2), dim=1)
            # output = torch.cat((output, output_for_aux), dim=1)
            output = torch.cat([output, output_for_aux, output_for_aux2], dim=1)
            return output, output_for_aux, obj_output
            # return output



class LeptonJetHeavyFlavourHead(nn.Module):
    """Predict HF for each lepton and pool its jet context for event classification."""

    context_dim = 48

    def __init__(self, embed_dim, jet_tag_indices):
        super().__init__()
        self.jet_tag_indices = jet_tag_indices
        self.lepton_proj = nn.Linear(embed_dim, 24)
        self.jet_proj = nn.Linear(embed_dim, 24)
        self.pair_embed = nn.Sequential(
            nn.Linear(53, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, self.context_dim), nn.GELU()
        )
        self.pair_rank = nn.Linear(self.context_dim, 1)
        self.no_jet = nn.Linear(embed_dim, 1)
        self.classifier = nn.Sequential(nn.Linear(embed_dim + self.context_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, tokens, features, vectors, valid):
        # tokens has shape [B, P, D] for B events, P object slots and D features per token.
        # The C raw features are [B, C, P], the four-vectors are [B, 4, P], and valid is [B, P].
        # In the matrix-run layout, slots 0–2 are leptons, 3–8 are central jets (unused slots
        # are masked), and slot 9 is MET. HF pairing uses only slots 0–8.
        # Update this head before changing the lepton/jet slot layout or including forward jets
        # as partners. These slices do not check object types and can silently misassign slots.
        # TODO: Remove the dependence on slot order. One option is to predict an association matrix
        # over all object pairs and use reconstructed object types to select eligible lepton–jet pairs.
        jets = slice(3, min(9, tokens.size(1) - 1))
        # With J central-jet slots, these slices give [B, 3, D] and [B, J, D], including padded slots.
        leptons, jet_tokens = tokens[:, :3], tokens[:, jets]

        vectors = vectors.float() * 0.001  # MeV to GeV
        # The None axes pair every lepton with every central jet. Adding [B, 4, 3, 1] and
        # [B, 4, 1, J] gives [B, 4, 3, J]. Taking the invariant mass leaves [B, 3, J].
        pair_mass = to_m2(vectors[:, :, :3, None] + vectors[:, :, None, jets], eps=0).squeeze(1).sqrt()
        pt = torch.linalg.vector_norm(vectors[:, :2], dim=1)
        pt_ratio = pt[:, :3, None] / pt[:, None, jets].clamp_min(1.e-4)
        eta, phi = features[:, 1], features[:, 2]
        dr = delta_r2(eta[:, :3, None], phi[:, :3, None], eta[:, None, jets], phi[:, None, jets]).sqrt()
        if self.jet_tag_indices is None:
            tag = available = torch.zeros_like(dr)
        else:
            quantile, known = self.jet_tag_indices
            tag = features[:, quantile, None, jets].expand_as(dr)
            available = features[:, known, None, jets].expand_as(dr)
        # Stack five numbers per pair to get [B, 3, J, 5]. Each jet's tag is shared across the lepton rows.
        kinematics = torch.stack((dr, pair_mass.log1p(), pt_ratio.clamp_max(50).log1p(), tag, available), dim=-1)

        # Expand [B, 3, 1, 24] and [B, 1, J, 24] to the same [B, 3, J, 24] pair grid.
        lepton_embed = self.lepton_proj(leptons).unsqueeze(2).expand(-1, -1, jet_tokens.size(1), -1)
        jet_embed = self.jet_proj(jet_tokens).unsqueeze(1).expand(-1, leptons.size(1), -1, -1)
        # Concatenate 24 lepton + 24 jet + 5 kinematic/tag features, then embed each pair in 48 dimensions.
        pairs = self.pair_embed(torch.cat((lepton_embed, jet_embed, kinematics), dim=-1))

        # ranks is [B, 3, J]. Append one learned "no jet" score per lepton. Each row of
        # probs then sums to one over J + 1 choices, with zero probability for padded jets.
        ranks = self.pair_rank(pairs).squeeze(-1).masked_fill(~valid[:, None, jets], -torch.inf)
        probs = torch.cat((ranks, self.no_jet(leptons)), dim=-1).softmax(dim=-1)
        # Use the J jet columns to weight the pair embeddings. Summing over jets gives [B, 3, 48].
        # An event with no valid central jet has zero jet context for each lepton.
        context = (probs[:, :, :-1, None] * pairs).sum(dim=2)
        # The classifier maps [B, 3, D + 48] to one HF logit per lepton [B, 3, 1].
        hf = self.classifier(torch.cat((leptons, context), dim=-1))

        # For the event classifier, give more weight to the jet context of leptons with higher HF logits.
        # This softmax runs over the three leptons. The binary object loss still receives each raw HF logit.
        valid_leptons = valid[:, :3, None]
        pool_logits = hf.masked_fill(~valid_leptons, torch.finfo(hf.dtype).min)
        # A finite minimum keeps softmax defined even for an empty row. Masking again gives it zero weight.
        lepton_weights = pool_logits.softmax(dim=1).masked_fill(~valid_leptons, 0.)
        event_jet_context = (lepton_weights * context).sum(dim=1)  # [B, 3, 48] -> [B, 48]

        # A logit of -20 gives a near-zero sigmoid probability for missing leptons and nonlepton slots.
        hf = hf.masked_fill(~valid_leptons, -20.)
        # pad reads axes from the end. (0, 0) keeps the single logit column.
        # (0, P - 3) appends rows after the three leptons, giving [B, P, 1] aligned with the input slots.
        hf = nn.functional.pad(hf, (0, 0, 0, tokens.size(1) - leptons.size(1)), value=-20.)
        return hf, event_jet_context


class ParticleTransformerSimple(_NormalizedTransformer):

    def __init__(self,
                 input_dim,
                 global_input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 global_embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 fc_params=[[64,0], [256,0], [64,0]],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 num_obj_classes=2,
                 lepton_jet_hf=None,
                 shared_parent_rank=None,
                 num_shared_parent_types=4,
                 shared_parent_feedback=None,
                 **kwargs) -> None:
        super().__init__(input_dim, global_input_dim, **kwargs)

        # Jet candidates and per-object truth IDs need the original slot ordering.
        use_lepton_heads = lepton_jet_hf is not None
        self.trimmer = SequenceTrimmer(
            enabled=trim and not for_inference and not use_lepton_heads and shared_parent_rank is None
        )
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        assert global_embed_dims[-1] == embed_dims[-1] if len(global_embed_dims) > 0 else input_dim, \
            'The last dimension of global_embed_dims must match the last dimension of embed_dims.'
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.pair_extra_dim = pair_extra_dim
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.global_embed = GlobalEmbed(global_input_dim, global_embed_dims, activation=activation) if len(global_embed_dims) > 0 else nn.Identity()
        self.blocks = nn.ModuleList([BlockGlobalFeat(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([BlockGlobalFeat(**cfg_cls_block) for _ in range(num_cls_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = 2*embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        self.num_obj_classes = num_obj_classes
        self.hf_head = None
        if use_lepton_heads:
            if num_obj_classes != 5 or input_dim < 3:
                raise ValueError("lepton_jet_hf requires five object logits and eta/phi in feature columns 1 and 2")
            tag_indices = lepton_jet_hf.get("jet_tag_indices")
            if tag_indices is not None and (len(tag_indices) != 2 or len(set(tag_indices)) != 2 or
                                            any(i < 0 or i >= input_dim for i in tag_indices)):
                raise ValueError("jet_tag_indices must give the tag quantile and availability feature columns")
            self.hf_head = LeptonJetHeavyFlavourHead(embed_dim, tag_indices)
        self.obj_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 4 if use_lepton_heads else self.num_obj_classes)
        )

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

        # Initialize the optional layer last so both configurations start with the same shared weights for a given seed.
        self.hf_event_projection = None
        if use_lepton_heads and self.fc is not None and lepton_jet_hf.get("use_event_context", False):
            # With no eligible lepton–jet pair the context is zero. Omitting the bias keeps its contribution zero.
            self.hf_event_projection = nn.Linear(self.hf_head.context_dim, 2 * embed_dim, bias=False)
        self.shared_parent_veto_indices = None
        feedback_heads = None
        if shared_parent_feedback is not None and shared_parent_feedback is not False:
            if shared_parent_rank is None or num_layers < 2:
                raise ValueError("shared_parent_feedback requires a shared-parent head and at least two object blocks")
            if not isinstance(shared_parent_feedback, dict):
                raise ValueError("Use model_config.model_params_for(data_config) to resolve the shared-parent veto features")
            # Older recipes stored the single MET column directly.
            indices = shared_parent_feedback.get("veto_feature_indices", [shared_parent_feedback.get("met_feature_index")])
            if any(not isinstance(i, int) or not 0 <= i < input_dim for i in indices):
                raise ValueError("Each shared-parent veto index must point to a flag in obj_feats")
            self.shared_parent_veto_indices = tuple(indices)
            feedback_heads = cfg_block['num_heads']
        self.shared_parent_head = None
        if shared_parent_rank is not None:
            self.shared_parent_head = SharedParentHead(embed_dim, shared_parent_rank, num_shared_parent_types, feedback_heads)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Earlier Synapse-dev checkpoints stored this head under "parent_head".
        if self.shared_parent_head is not None:
            for name in self.shared_parent_head.state_dict():
                old = prefix + "parent_head." + name
                new = prefix + "shared_parent_head." + name
                if old in state_dict and new not in state_dict:
                    state_dict[new] = state_dict.pop(old)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, g, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # g: (N, C) -- global feature
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        raw_features = x
        x, g = self.normalize_inputs(x, g)

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            g = self.global_embed(g)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu).flatten(0, 1)  # (N*num_heads, P, P)

            shared_parent_logits = None
            for index, block in enumerate(self.blocks):
                if self.shared_parent_veto_indices is not None and index == len(self.blocks) - 1:
                    # Use the penultimate block's predictions for both the pair loss and the last block's attention.
                    shared_parent_logits = self.shared_parent_head(x.transpose(0, 1))  # [batch, parent, object, object]
                    veto_flags = (raw_features[:, i] for i in self.shared_parent_veto_indices)
                    eligible = shared_parent_object_mask(~padding_mask, veto_flags)
                    shared_parent_bias = self.shared_parent_head.attention_bias(shared_parent_logits, eligible)
                    attn_mask = shared_parent_bias if attn_mask is None else attn_mask + shared_parent_bias
                x = block(x, x_cls=None, x_glob=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            g = g.expand(1, x.size(1), -1) # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, x_glob=g, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0) # (N, C)
            g = g.squeeze(0)
            x_cls = torch.cat((x_cls, g), dim=1)
            # fc
            if self.fc is None:
                return x_cls
            tokens = x.transpose(0, 1)
            obj_output = self.obj_fc(tokens)
            if self.hf_head is not None:
                if v is None:
                    raise ValueError("The lepton–jet heavy-flavour head requires object four-vectors")
                hf, event_jet_context = self.hf_head(tokens, raw_features, v, mask.squeeze(1))
                obj_output = torch.cat((obj_output, hf), dim=-1)
                if self.hf_event_projection is not None:
                    # Add the pooled jet context to the [CLS, global] features before predicting the event class.
                    x_cls = x_cls + self.hf_event_projection(event_jet_context)
            output = self.fc(x_cls)
            if self.shared_parent_head is not None and shared_parent_logits is None:
                # Recipes using the matrices only for supervision predict them from the final tokens.
                shared_parent_logits = self.shared_parent_head(tokens)
            aux = {"shared_parent_logits": shared_parent_logits} if shared_parent_logits is not None else None
            return output, aux, obj_output.reshape(-1, self.num_obj_classes)

