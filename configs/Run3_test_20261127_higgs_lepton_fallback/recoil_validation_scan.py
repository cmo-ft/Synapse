"""Vary only the weight of the frozen model's recoil contribution on full validation MC."""
import argparse
from contextlib import ExitStack
import csv
import json

import awkward as ak
import numpy as np
import torch

from paths import RUN
from probe_higgs_feedback import INPUTS, open_tree, raw_inputs, read_checkpoint
from process_catalog import GROUPS, process_codes
from synapse.core.tools import extract_fields_from_expr
from synapse.core.working_point_significance import working_point_significance

SCALES = (0., .5, 1., 1.5, 2.)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=rows[0])
        out.writeheader()
        out.writerows(rows)


@torch.inference_mode()
def score_validation(fold, folder):
    cfg, model, _, checkpoint = read_checkpoint(folder)
    def capture(module, args):
        model_input[0] = args[0]
    model_input = [None]
    model.fc.register_forward_pre_hook(capture)
    features = {name for group in INPUTS for name in cfg.inputs[group]}
    expr = {k: val for k,val in cfg.new_variables.items() if k in features}
    fields = features - expr.keys() | set().union(*(extract_fields_from_expr(v) for v in expr.values()))
    with ExitStack() as stack:
        raw = open_tree(stack, RUN/f"dataset/3fold_{(fold+1)%3}.root")
        t = open_tree(stack, folder/"validation/validation_scores.root")
        data = t.arrays(["eventNumber","dsid","class_id","n_jet","weight","p_signal","p_extconv_bkg"],library="np")
        if t.num_entries != raw.num_entries:
            raise ValueError("Validation row counts differ")
        result = {s:np.empty((t.num_entries,2), dtype="f4") for s in SCALES}
        for start in range(0,t.num_entries,16000):
            stop = min(start+16000,t.num_entries)
            chunk = raw.arrays([*sorted(fields),"eventNumber","dsid"],entry_start=start,entry_stop=stop,library="ak")
            for name in ("eventNumber","dsid"):
                if not np.array_equal(ak.to_numpy(chunk[name]),data[name][start:stop]):
                    raise ValueError("Different validation row identities")
            inputs = raw_inputs(chunk,cfg,expr)
            for ib in range(0,len(chunk),384):
                n = min(384,len(chunk)-ib)
                batch = [x[ib:ib+n] for x in inputs]
                logits,aux,_ = model(*batch)
                native = model_input[0]
                recoil = model.recoil_projection(aux["wz_recoil"].tanh().unsqueeze(1))
                for scale in SCALES:
                    v = logits if scale == 1 else model.fc(native+(scale-1)*recoil)
                    result[scale][start+ib:start+ib+n] = v.softmax(1)[:,(0,2)].numpy()
    return checkpoint,data,result


def analyze(fold):
    folder = RUN/f"output/fold_{fold}"
    checkpoint,data,probs = score_validation(fold,folder)
    out = folder/"evaluation/ancestry_comparison"
    pre = data["n_jet"]>=1
    hh = data["class_id"]==0
    proc = process_codes(data["dsid"],RUN/"dataset/summary.json")
    w = data["weight"].astype("f8")
    pos_bkg = pre & ~hh & (w>0)
    if not np.array_equal(hh,proc==-1):
        raise ValueError("Signal class and DSID disagree")
    nominal = np.stack([data["p_signal"],data["p_extconv_bkg"]],axis=1)
    q99 = np.quantile(np.abs(probs[1.]-nominal), .99, axis=0)
    if q99.max()>.005:
        raise ValueError(f"Native CPU application differs from stored validation scores: {q99}")
    comparison = list(probs.items())+[("stored_gpu",nominal)]
    selections,metrics,rows = {},{},[]
    for scale,p in comparison:
        if not np.isfinite(p).all():
            raise ValueError("Nonfinite scores")
        met = working_point_significance(p[:,0],hh,w,pre,p[:,1]<.8)
        selected = [pre & (p[:,1]<.8) & (p[:,0]>=cut) for cut in met["cuts"]]
        met["positive_R20"] = float(w[pos_bkg].sum()/w[selected[0]&pos_bkg].sum())
        metrics[str(scale)] = met
        selections[scale] = selected
    for scale,sels in selections.items():
        for wp,sel,base in zip((20,30,50),sels,selections[1.]):
            for idx,group in [(-2,"all backgrounds"),(-1,"HH"),*enumerate(GROUPS.values())]:
                group_mask = ~hh if idx==-2 else proc==idx
                switch = group_mask & (sel!=base)
                prev,cur = w[group_mask & base].sum(),w[group_mask & sel].sum()
                rows.append(dict(recoil_scale=scale,working_point=wp,process=group,signed_yield=float(cur),
                                 delta_from_native_cpu=float(cur-prev),
                                 paired_mc_error=float(np.sqrt(np.square(w[switch]).sum()))))
    report = dict(fold=fold,split="validation",checkpoint=checkpoint,entries=len(w),
                  q99_native_vs_saved_probability_difference=q99.tolist(),metrics=metrics)
    (out/"recoil_strength_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    save_csv(out/"recoil_strength_validation_yields.csv",rows)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    analyze(args.fold)
