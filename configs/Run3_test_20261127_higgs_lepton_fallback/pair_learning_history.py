"""Check when within-event HH Higgs pairing emerges in a fixed fold-0 validation sample."""
import argparse
import csv
from hashlib import sha256
import json

import awkward as ak
import numpy as np
import torch
from paths import RUN, ANCESTRY_RUN, TREE_RUN
from probe_higgs_feedback import INPUTS, raw_inputs, read_checkpoint
from compare_event_scores import open_root
from compare_higgs_supervision import I, J, PAIR
from synapse.core.config import DataConfig
from synapse.core.tools import extract_fields_from_expr


def validation_rows(fold, cfg, sample_size):
    features = {name for group in INPUTS for name in cfg.inputs[group]}
    expr = {k:v for k,v in cfg.new_variables.items() if k in features}
    fields = features - expr.keys() | set().union(*(extract_fields_from_expr(v) for v in expr.values()))
    fields |= {"dsid","eventNumber","weight","valid","isMET","higgsPairKnown","higgsPairSame"}
    inputs = [[] for _ in INPUTS]
    pair_masks, same_masks, weights = [], [], []
    digest = sha256()
    with open_root(TREE_RUN/f"dataset/3fold_{(fold+1)%3}.root") as f:
        tree=f["tree"]
        hh=np.flatnonzero(tree["class_id"].array(library="np")==0)
        rows=np.sort(np.random.default_rng(7820+fold).choice(hh,size=sample_size,replace=False))
        for start in range(0,tree.num_entries,20000):
            stop=min(start+20000,tree.num_entries)
            take=rows[np.searchsorted(rows,start):np.searchsorted(rows,stop)]-start
            if not len(take):
                continue
            chunk=tree.arrays(sorted(fields),entry_start=start,entry_stop=stop,library="ak")[take]
            for output, x in zip(inputs,raw_inputs(chunk,cfg,expr)):
                output.append(x)
            for key in ("dsid","eventNumber"):
                digest.update(ak.to_numpy(chunk[key]).tobytes())
            physical=ak.to_numpy(chunk["valid"]).astype(bool) & ~ak.to_numpy(chunk["isMET"]).astype(bool)
            pair_masks.append(physical[:,I] & physical[:,J] & (((ak.to_numpy(chunk["higgsPairKnown"])[:,I]>>J)&1)!=0))
            same_masks.append(((ak.to_numpy(chunk["higgsPairSame"])[:,I]>>J)&1)!=0)
            weights.append(np.abs(ak.to_numpy(chunk["weight"]).astype("f8")))
    return [torch.cat(x) for x in inputs], np.concatenate(pair_masks), np.concatenate(same_masks), np.concatenate(weights), digest.hexdigest()


@torch.inference_mode()
def evaluate(fold):
    out=RUN/f"output/fold_{fold}/evaluation/ancestry_comparison"
    cfg=DataConfig(str(RUN/f"output/fold_{fold}/test/data_config.yaml"))
    inputs,known,same,w,digest=validation_rows(fold,cfg,16000)
    rows=[]
    for name,run,epochs in (("previous",ANCESTRY_RUN,(10,20,24,27,31,35)),
                            ("current",RUN,(10,20,24,27,31))):
        folder=run/f"output/fold_{fold}"
        for epoch in epochs:
            paths=list((folder/"fit/checkpoints").glob(f"model-epoch=epoch={epoch-1}-*.ckpt"))
            if len(paths)!=1:
                raise ValueError(f"Unexpected checkpoint count for {name} {epoch}")
            _,model,channel,_=read_checkpoint(folder,paths[0])
            logits=[]
            for start in range(0,len(w),384):
                aux=model(*(x[start:start+384] for x in inputs))[1]
                logits.append(aux["shared_parent_logits"][:,channel,I,J].numpy())
            pred=np.concatenate(logits)
            for kind, cols in PAIR.items():
                candidate=known & cols
                truth=candidate & same
                usable=truth.any(1) & (candidate & ~same).any(1)
                best=np.argmax(np.where(candidate[usable],pred[usable],-np.inf),axis=1)
                correct=truth[usable][np.arange(len(best)), best]
                wt=w[usable]
                accuracy=np.average(correct,weights=wt)
                error=np.linalg.norm(wt*(correct-accuracy))/wt.sum()
                rows.append(dict(sample="validation HH", model=name,epoch=epoch,objects=kind,
                                 eligible_events=int(usable.sum()),accuracy=float(accuracy),mc_error=float(error)))
    with (out/"pair_learning_history_validation.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (out/"pair_learning_history_sample.json").write_text(json.dumps(dict(fold=fold,seed=7820+fold,
        selected_HH=len(w),dsid_eventnumber_sha256=digest,reference=TREE_RUN.name), indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold",required=True,type=int,choices=(0,))
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    evaluate(args.fold)
