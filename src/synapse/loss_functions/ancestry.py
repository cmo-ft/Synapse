"""Auxiliary training on verified ancestry, hadronic W assignments and WZ control events."""
import numpy as np
import torch
from torch.nn import functional as F
from .lepton_objects import EventAndLeptonObjectLoss, shared_parent_bce


def balanced_w_assignment(logits, ids):
    """Half the jet-pair weight goes to the true pair when wrong central pairs exist."""
    i,j = torch.triu_indices(ids.size(1),ids.size(1),offset=1,device=logits.device)
    a,b = ids[:,i], ids[:,j]
    known = (a>=0) & (b>=0)
    same = (a>0) & (a==b)
    jj = (i>=3) & (j<9)
    cost = F.binary_cross_entropy_with_logits(logits[:,i,j],same.to(logits.dtype),reduction="none")
    def average(mask):
        return cost.masked_fill(~mask,0).sum(1) / mask.sum(1).clamp_min(1)
    pos,neg = known & jj & same, known & jj & ~same
    wrong = neg.any(1)
    jets = torch.where(wrong, (average(pos)+average(neg))/2, average(pos))
    other = known & ~jj
    # Retain the original lepton-pair and lepton-jet supervision without letting
    # the number of easy combinations set the weight of the hadronic assignment.
    return torch.where(other.any(1),(jets+average(other))/2,jets)


class AncestryControlLoss(EventAndLeptonObjectLoss):
    def __init__(self, ancestry_branches, ancestry_loss_weight=0.02, control_z_weight=0.10,
                 recoil_weight=0.10, **kwargs):
        super().__init__(**kwargs)
        self.ancestry_branches = tuple(ancestry_branches)
        self.ancestry_loss_weight = ancestry_loss_weight
        self.control_z_weight = control_z_weight
        self.recoil_weight = recoil_weight
        self.w_index = self.shared_parent_branches.index("relationW")
        self.z_index = self.shared_parent_branches.index("relationZ")
        self.register_buffer("base_fraction",torch.tensor(1.))
        self.register_buffer("control_scale",torch.ones(2))
        self.register_buffer("ancestry_scale",torch.ones(len(self.ancestry_branches)))

    @torch.no_grad()
    def prepare_training_fold(self, targets, mask, event_weights):
        super().prepare_training_fold(targets,mask,event_weights)
        info=targets["physics_info"]
        control=info[:,0].astype(bool)
        if not control.any() or control.all() or np.any(event_weights[control]!=0):
            raise ValueError("Expected both base and control events, with zero main weight on controls")
        self.base_fraction.fill_(float((~control).mean()))
        totals=np.zeros(len(self.ancestry_branches))
        zsum=0.
        for start in range(0,len(info),16000):
            sl=slice(start,start+16000)
            valid=mask[sl,0].astype(bool) & ~targets["isMET"][sl].astype(bool)
            wt=event_weights[sl].astype("f8")
            for k,key in enumerate(self.ancestry_branches):
                totals[k]+=wt[((targets[key][sl]>=0)&valid).any(1)].sum()
            has_z=(((targets["relationZ"][sl]>=0)&valid).sum(1)>=2)
            zsum+=info[sl,1][control[sl]&has_z].sum(dtype="f8")
        recoil_sum=info[:,1][control].sum(dtype="f8")
        if np.any(totals<=0) or min(zsum,recoil_sum)<=0:
            raise ValueError(f"Missing ancestry, Z or recoil supervision: {totals}, {zsum}, {recoil_sum}")
        self.ancestry_scale.copy_(torch.as_tensor(len(info)/totals))
        self.control_scale.copy_(torch.tensor([len(info)/zsum,len(info)/recoil_sum]))

    def parent_losses(self, logits, ids, physics_info):
        parts=list(shared_parent_bce(logits,ids).unbind(1))
        iw=self.w_index
        balanced=balanced_w_assignment(logits[:,iw],ids[:,iw])
        parts[iw]=torch.where(physics_info[:,5].bool(),balanced,parts[iw])
        return torch.stack(parts,1)

    def forward(self,pred,target,weight,mean_train_weight=None,physics_info=None,
                object_targets=None,object_mask=None,**kwargs):
        if not self.training:
            return super().forward(pred,target,weight,mean_train_weight/self.base_fraction)
        loss,scores=super().forward(pred,target,weight,mean_train_weight,object_targets=object_targets,
                                   object_mask=object_mask,physics_info=physics_info,**kwargs)
        valid=object_mask[:,0].bool() & ~object_targets["isMET"].bool()
        y=torch.stack([object_targets[k] for k in self.ancestry_branches],-1)
        known=(y>=0)&valid.unsqueeze(-1)
        pred_a=pred[1]["ancestry_logits"]
        costs=F.binary_cross_entropy_with_logits(pred_a,(y==1).to(pred_a.dtype),reduction="none")
        per_event=costs.masked_fill(~known,0).sum(1)/known.sum(1).clamp_min(1)
        loss=loss+self.ancestry_loss_weight*(weight*(per_event*self.ancestry_scale).sum(1)).mean()
        control_weight=physics_info[:,0]*physics_info[:,1]
        zid=object_targets["relationZ"].masked_fill(~valid,-1).unsqueeze(1)
        z=shared_parent_bce(pred[1]["shared_parent_logits"][:,self.z_index:self.z_index+1],zid)[:,0]
        cr=F.binary_cross_entropy_with_logits(pred[1]["wz_recoil"],physics_info[:,4],reduction="none")
        return loss+(control_weight*(self.control_z_weight*self.control_scale[0]*z+
                                    self.recoil_weight*self.control_scale[1]*cr)).mean(),scores
