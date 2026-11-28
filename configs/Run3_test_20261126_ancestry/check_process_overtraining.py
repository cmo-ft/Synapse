"""Compare the same Synapse event loss on training, validation and test MC at saved checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import awkward as ak
import numpy as np
import torch
import uproot


PROCESSES=("singleH","VVV")
SPLITS=("training","validation","test")


def select(weight,groups,names,rng,target):
    """Probability proportional to absolute MC weight, with the largest weights taken with certainty."""
    prob=np.zeros_like(weight,dtype="f8")
    for group in PROCESSES:
        rows=np.flatnonzero((groups==names.index(group)) & (weight>0))
        w=weight[rows]
        if len(w)<=target:
            prob[rows]=1
            continue
        low,high=0.,target/w.sum()
        while np.minimum(1.,w*high).sum()<target:
            high*=2
        for _ in range(45):
            mid=(low+high)/2
            if np.minimum(1.,w*mid).sum()<target: low=mid
            else: high=mid
        prob[rows]=np.minimum(1.,w*high)
    return prob, rng.random(len(weight))<prob


def read_events(run, fold, split, rem, cfg, target, seed):
    from process_catalog import GROUPS,process_codes
    from synapse.core.dataset import prepare_data
    from synapse.core.tools import extract_fields_from_expr
    names=list(GROUPS.values())
    path=run/"dataset"/f"3fold_{rem}.root"
    with uproot.open(path) as f:
        tree=f["tree"]
        a=tree.arrays(["dsid","eventNumber","class_id","n_jet","weight"],library="np")
        if not np.all(a["eventNumber"]%3==rem): raise ValueError(f"Wrong events in {path}")
        groups=process_codes(a["dsid"],run/"dataset/summary.json")
        absolute=np.abs(a["weight"].astype("f8"))*(a["n_jet"]>=1)
        prob,keep=select(absolute,groups,names,np.random.default_rng(seed),target)
        branches=set(cfg.active_keys)
        for expr in cfg.new_variables.values(): branches.update(extract_fields_from_expr(expr))
        fields=[]
        for start in range(0,len(keep),24000):
            use=keep[start:start+24000]
            if use.any():
                raw=tree.arrays(filter_name=list(branches),entry_start=start,entry_stop=start+len(use))
                fields.append(raw[use])
        features=prepare_data(ak.concatenate(fields),cfg,"val")
    weight=features[cfg.weight_key].astype("f8")/prob[keep]
    y=features["_class_label"]
    np.testing.assert_array_equal(y,a["class_id"][keep])
    reference=None
    if split!="training":
        with uproot.open(run/"output"/f"fold_{fold}"/split/(split+"_scores.root")) as f:
            ref=f["tree"].arrays(["eventNumber","dsid","p_signal","p_bkg","p_extconv_bkg"],library="np")
        for key in ("eventNumber","dsid"):
            np.testing.assert_array_equal(ref[key],a[key])
        reference=np.column_stack([ref[k][keep] for k in ("p_signal","p_bkg","p_extconv_bkg")])
    inputs=[torch.from_numpy(features[k]) for k in ("obj_feats","evt_feats","obj_4vecs","mask")]
    eventkeys=np.rec.fromarrays((a["dsid"][keep],a["eventNumber"][keep]), names="dsid,event")
    return dict(inputs=inputs,y=torch.from_numpy(y),weight=weight,keys=eventkeys,reference=reference,
                category=np.asarray(names)[groups[keep]],loss={})


def weight_mean(x,w,keys):
    value=np.average(x,weights=w)
    i=np.unique(keys,return_inverse=True)[1].ravel()
    influence=np.bincount(i,weights=w*(x-value))/w.sum()
    return float(value),float(np.linalg.norm(influence))


def main(run,fold,target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(run/"frozen"))
    sys.path.insert(0,str(run/"frozen/plots"))
    from synapse.core.config import DataConfig,ModelConfig
    from synapse.models.ParticleTransformer import ParticleTransformerSimple
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    uproot.open.defaults["handler"]=uproot.source.file.MemmapSource
    folder=run/"output"/f"fold_{fold}"
    hist=[json.loads(p.read_text()) for p in sorted((folder/"fit/epochs").glob("epoch_*.json"))]
    minimum=min(hist,key=lambda h:h["val_loss_epoch"])["epoch"]
    best=max(hist,key=lambda h:h["val_Significance20_30_50_epoch"])["epoch"]
    epochs=sorted({15,minimum,best,hist[-1]["epoch"]})
    cfg=DataConfig(str(folder/"validation/data_config.yaml"))
    model_cfg=ModelConfig(str(folder/"validation/model_config.yaml"))
    params=model_cfg.model_params_for(cfg)
    data={}
    for split,rem in zip(SPLITS,((fold+2)%3,(fold+1)%3,fold)):
        data[split]=read_events(run,fold,split,rem,cfg,target,301127+rem)
        print(split,"sampled",len(data[split]["weight"]),flush=True)
    closure={}
    loss_closure={}
    with torch.inference_mode():
        for epoch in epochs:
            path,=(folder/"fit/checkpoints").glob(f"model-epoch=epoch={epoch-1}-*.ckpt")
            model=ParticleTransformerSimple(**params,for_inference=True)
            state=torch.load(path,map_location="cpu",weights_only=False)["state_dict"]
            model.load_state_dict({k.removeprefix("model."):v for k,v in state.items() if k.startswith("model.")},strict=True)
            model.eval()
            for split,d in data.items():
                losses,prob=[],[]
                for a in range(0,len(d["y"]),192):
                    sl=slice(a,a+192)
                    logits=model(*(v[sl] for v in d["inputs"]))[0]
                    losses.append(torch.nn.functional.cross_entropy(logits,d["y"][sl],reduction="none").numpy())
                    if epoch==best and d["reference"] is not None:
                        prob.append(logits.softmax(1).numpy())
                d["loss"][epoch]=np.concatenate(losses).astype("f8")
                if prob:
                    err=np.abs(np.concatenate(prob)-d["reference"])
                    closure[split]=np.quantile(err,[.5,.99,1]).tolist()
                    class_ids=d["y"].numpy()
                    native=d["reference"][np.arange(len(class_ids)), class_ids].astype("f8")
                    if not np.all((native>0) & (native<=1)):
                        raise ValueError(f"Invalid production probabilities for {split}")
                    native_loss=-np.log(native)
                    loss_closure[split]={}
                    for group in PROCESSES:
                        mask=d["category"]==group
                        weight,keys=d["weight"][mask],d["keys"][mask]
                        reference,mc_se=weight_mean(native_loss[mask],weight,keys)
                        delta=d["loss"][epoch][mask]-native_loss[mask]
                        bias=float(np.average(delta,weights=weight))
                        mae=float(np.average(np.abs(delta),weights=weight))
                        loss_closure[split][group]=dict(native_ce=reference,mc_se=mc_se,
                                                         cpu_minus_native=bias,mean_absolute_error=mae)
                        if abs(bias)>.10*mc_se or mae>.30*mc_se:
                            raise ValueError(f"CPU and production loss differ for {split} {group}: "
                                             f"{loss_closure[split][group]}")
                    print("CPU versus published process means",split,loss_closure[split],flush=True)
            print("Evaluated checkpoint",epoch,flush=True)
    rows=[]
    for split,d in data.items():
        for g in PROCESSES:
            use=d["category"]==g
            w,keys=d["weight"][use],d["keys"][use]
            for epoch in epochs:
                mean,err=weight_mean(d["loss"][epoch][use],w,keys)
                delta,ed=weight_mean((d["loss"][epoch]-d["loss"][minimum])[use],w,keys)
                rows.append(dict(split=split,process=g,epoch=epoch,events=int(use.sum()),
                                 weighted_event_ce=mean,mc_se=err,change_from_min_val_epoch=delta,paired_change_se=ed))
    out=folder/"evaluation/validation_test"
    out.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out) as tmp:
        stage=Path(tmp)
        with (stage/"synapse_process_loss.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)
        details=dict(fold=fold,checkpoints=epochs,minimum_validation_loss_epoch=minimum,production_epoch=best,
                     selection="Base SR and ExtConvCR MC, n_jet >= 1, before the classifier veto.",
                     estimator="Mean three-class event cross-entropy within each process, weighted by the training-fold class factors times absolute nominal MC weight divided by the event sampling probability. The same rows are evaluated at every checkpoint. Error bars are approximate event-key-clustered MC standard errors.",
                     expected_sample_per_process_per_split=target,seed_by_remainder=[301127+i for i in range(3)],
                     production_score_abs_error_median_p99_max=closure,
                     production_loss_by_split_and_process=loss_closure,
                     precision_check="All epochs use the same CPU FP32 implementation. At the production checkpoint, each mean must agree with the stored GPU scores within 0.1 MC standard error. The weighted mean absolute event-loss difference must be below 0.3 MC standard error.")
        (stage/"synapse_process_loss_settings.json").write_text(json.dumps(details,indent=2)+"\n")
        fig,axes=plt.subplots(1,2,figsize=(9.8,4.9))
        for ax,g in zip(axes,PROCESSES):
            for split,color,offset in zip(SPLITS,("#595B65","#266D9C","#C15923"),(-.35,0,.35)):
                points=[r for r in rows if r["split"]==split and r["process"]==g]
                ax.errorbar([p["epoch"]+offset for p in points],[p["weighted_event_ce"] for p in points],
                            yerr=[p["mc_se"] for p in points],fmt="o-",color=color,lw=1.6,capsize=3,label=split.capitalize())
            ax.axvline(best,color=".6",ls=":",lw=1)
            ax.set_title("Single H" if g=="singleH" else "VVV",loc="left")
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            ax.grid(axis="y",alpha=.2)
            ax.spines[["right","top"]].set_visible(False)
        axes[0].set_ylabel("Mean event cross-entropy")
        fig.suptitle(f"Fold {fold}  ·  Synapse event loss on identical rows across checkpoints",x=.07,ha="left",fontsize=14)
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc="upper left",frameon=False,bbox_to_anchor=(.06,.92))
        fig.subplots_adjust(left=.09,right=.98,top=.76,bottom=.19,wspace=.23)
        fig.text(.09,.035,"All three samples use the same loss and MC weighting. Dotted line marks the production checkpoint. Error bars show one MC standard error.",fontsize=8)
        fig.savefig(stage/"synapse_process_loss.pdf")
        plt.close(fig)
        for p in stage.iterdir(): p.replace(out/p.name)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--fold",type=int,required=True)
    p.add_argument("--target",type=int,default=6500)
    args=p.parse_args()
    main(args.run.resolve(),args.fold,args.target)
