import unittest
import numpy as np
import torch

from synapse.core.working_point_significance import working_point_significance
from synapse.loss_functions.ancestry import balanced_w_assignment, AncestryControlLoss
from synapse.models.ParticleTransformer import ParticleTransformerSimple
from synapse.models.wz_recoil import forward_with_control


class AncestryTrainingTests(unittest.TestCase):
    def test_w_assignment_splits_weight_between_true_and_competing_jets(self):
        ids = torch.tensor([[1,2,3,4,4,0,0,-1,-1,-1,-1,-1]])
        logits = torch.zeros(1,12,12,requires_grad=True)
        balanced_w_assignment(logits,ids).sum().backward()
        g = logits.grad[0]
        i,j = torch.triu_indices(4,4,offset=1)+3
        self.assertAlmostEqual(float(-g[3,4]),0.125)
        self.assertAlmostEqual(float(g[i,j][g[i,j]>0].sum()),0.125)
        self.assertAlmostEqual(float(g.tril().abs().sum()),0.)

    def test_control_updates_only_z_projection_and_recoil(self):
        torch.manual_seed(42)
        model=ParticleTransformerSimple(input_dim=14,global_input_dim=3,num_classes=3,
            embed_dims=[16],global_embed_dims=[16],pair_embed_dims=[8],num_heads=2,num_layers=5,
            num_cls_layers=1,fc_params=[[8,0]],num_obj_classes=5,trim=False,shared_parent_rank=2,
            shared_parent_feedback={"veto_feature_indices":[6]},ancestry_outputs=list("htwz"),
            global_input_batchnorm=False,standardization_clip=12,wz_recoil=True)
        model.setup_input_norm(torch.zeros(14),torch.ones(14),torch.zeros(3),torch.ones(3))
        x=torch.randn(4,14,12)
        x[:,4,:3]=torch.tensor([11,-11,13]); x[:,6]=0; x[:,6,9]=1
        x[:,7]=0; x[:,7,3:9]=1
        v=torch.randn(4,4,12)*1000
        v[:,3]=v[:,:3].norm(dim=1)+5
        mask=torch.ones(4,1,12,dtype=torch.bool)
        target={k:torch.full((4,12),-1,dtype=torch.int8) for k in
                ("relationHiggs","relationTop","relationW","relationZ","aH","aT","aW","aZ")}
        for k in ("partOrigin","isConv","isLepton","isHF","isMET"):
            target[k]=torch.zeros(4,12,dtype=torch.int8)
        target["isMET"][:,9]=1; target["isLepton"][:,:3]=1
        target["relationZ"][:,:3]=torch.tensor([1,1,0])
        for k in ("aH","aT","aW","aZ"):
            target[k][:,:3]=torch.tensor([1,0,-1])
        branches=["relation"+p for p in ("Higgs","Top","W","Z")]
        criterion=AncestryControlLoss(ancestry_branches=["aH","aT","aW","aZ"],shared_parent_branches=branches,
                                     balance_origin_classes=False,shared_parent_loss_weight=.02)
        info=torch.tensor([[0,1,3,1,0,0],[0,1,3,1,1,0],[1,1,3,1,1,0],[1,1,3,1,0,0.]],dtype=torch.float32)
        fold={k:v.numpy() for k,v in target.items()}
        fold["physics_info"]=info.numpy()
        criterion.prepare_training_fold(fold,mask.numpy(),np.array([1.,1.,0.,0.]))
        stats={k:q.clone() for k,q in model.named_buffers() if "running_" in k}
        inputs=[x[2:],torch.randn(2,3),v[2:],mask[2:]]
        pred=forward_with_control(model,inputs,torch.ones(2,dtype=torch.bool))
        cr_targets={k:q[2:] for k,q in target.items()}
        loss,_=criterion(pred,torch.tensor([1,1]),torch.zeros(2),0.5,
                         physics_info=info[2:],object_mask=mask[2:],object_targets=cr_targets)
        loss.backward()
        for k,q in model.named_buffers():
            if k in stats: torch.testing.assert_close(q,stats[k])
        for name,param in model.named_parameters():
            if name.startswith(("wz_recoil.","shared_parent_head.")): continue
            if param.grad is not None:
                self.assertEqual(float(param.grad.abs().sum()),0.,name)
        self.assertGreater(float(model.shared_parent_head.intercept.grad[3].abs()),0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.wz_recoil.parameters()))
        self.assertEqual(float(model.shared_parent_head.intercept.grad[:3].abs().sum()),0.)

    def test_significance_uses_disjoint_signal_bands(self):
        s=np.r_[np.arange(10,0,-1),np.arange(10,0,-1)]/10
        hh=np.repeat([True,False],10)
        w=np.r_[np.ones(10),np.array([2,-.1,2,2,2,2,2,2,2,2])]
        result=working_point_significance(s,hh,w,np.ones(20,bool),np.ones(20,bool))
        np.testing.assert_allclose([r["signal"] for r in result["exclusive"]],[2,1,2])
        self.assertAlmostEqual(result["exclusive"][0]["background_mc_variance"],4.01)
