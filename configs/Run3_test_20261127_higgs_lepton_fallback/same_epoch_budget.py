"""Reapply the previous best within the current fold's training budget on one CPU."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import awkward as ak
import numpy as np
import torch
import uproot

from paths import ANCESTRY_RUN, RUN, REPO
from compare_event_scores import bin_indices, changes, open_root, paired_bootstrap, score_path
from probe_higgs_feedback import INPUTS, raw_inputs, read_checkpoint
from process_catalog import GROUPS, process_codes
from synapse.core.tools import extract_fields_from_expr
from synapse.core.working_point_significance import working_point_significance

FIELDS = ("eventNumber","dsid","weight","label_signal","n_jet","evenet_pS","evenet_pConv","evenet_valid","xgb_score")
LCG = "/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"


def publish(path, result):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


@torch.inference_mode()
def apply(model, config, fold, split, temp):
    features = {name for group in INPUTS for name in config.inputs[group]}
    expr = {k:v for k,v in config.new_variables.items() if k in features}
    fields = features - expr.keys() | set().union(*(extract_fields_from_expr(v) for v in expr.values()))
    remainder = fold if split == "test" else (fold+1)%3
    with open_root(ANCESTRY_RUN/f"dataset/3fold_{remainder}.root") as source, open_root(score_path(RUN,fold,split)) as ref, uproot.recreate(temp) as dest:
        raw, current = source["tree"], ref["tree"]
        if raw.num_entries != current.num_entries:
            raise ValueError("Row count differs")
        tree = None
        for start in range(0, raw.num_entries, 16000):
            stop = min(start+16000, raw.num_entries)
            chunk = raw.arrays([*sorted(fields),"dsid","eventNumber"],entry_start=start,entry_stop=stop,library="ak")
            a = current.arrays(FIELDS,entry_start=start,entry_stop=stop,library="np")
            if not all(np.array_equal(ak.to_numpy(chunk[k]),a[k]) for k in ("eventNumber","dsid")):
                raise ValueError("Different current and previous row identities")
            columns = raw_inputs(chunk,config,expr)
            a["p_signal"], a["p_extconv_bkg"] = (np.empty(len(chunk),dtype="f4") for _ in (0,1))
            for idx in range(0,len(chunk),384):
                sl = slice(idx, min(idx+384,len(chunk)))
                logits = model(*(x[sl] for x in columns))[0]
                probs = logits.softmax(1).numpy()
                a["p_signal"][sl],a["p_extconv_bkg"][sl]=probs[:,0],probs[:,2]
            if tree is None:
                tree=dest.mktree("tree",{k:v.dtype for k,v in a.items()})
            tree.extend(a)
    print(f"{split} event scores complete",flush=True)


def compare(fold,split,scored,cuts=None):
    keys = ["p_signal","p_extconv_bkg","eventNumber","dsid","weight","n_jet","label_signal"]
    with open_root(scored) as old, open_root(score_path(RUN,fold,split)) as current:
        a,b = [f["tree"].arrays(keys,library="np") for f in (old,current)]
    if any(not np.array_equal(a[k],b[k]) for k in keys[2:]):
        raise ValueError("Scored event identity or weights changed")
    w,hh,pre = a["weight"].astype("f8"),a["label_signal"].astype(bool),a["n_jet"]>=1
    group = process_codes(a["dsid"],RUN/"dataset/summary.json")
    metrics,bins={},{}
    for name,x in (("previous_budget",a),("current",b)):
        score,veto=x["p_signal"],x["p_extconv_bkg"]<.8
        m=working_point_significance(score,hh,w,pre,veto,cuts[name] if cuts else None)
        bins[name] = bin_indices(score,pre & veto,m["cuts"])
        z = pre & ~hh & (w>0)
        m["R20"] = float(w[z].sum()/w[z & (bins[name]==0)].sum())
        metrics[name] = m
    base,new=bins["previous_budget"],bins["current"]
    metrics["paired_delta_z"]=paired_bootstrap(w,hh,base,new,56110+fold)
    rows=[]
    for k,wp in enumerate((20,30,50)):
        for idx,process in [(-2,"all backgrounds"),(-1,"HH"),*enumerate(GROUPS.values())]:
            mask=~hh if idx==-2 else group==idx
            rows.append(dict(split=split,cuts="validation" if cuts else "split",wp=wp,process=process,
                             **changes(w,mask,base<=k,new<=k)))
    print(split,"frozen validation cuts" if cuts else "equal efficiency",[(k,metrics[k]["quadrature"]) for k in ("previous_budget","current")],flush=True)
    return metrics,rows


def main(fold):
    out=RUN/f"output/fold_{fold}/evaluation/ancestry_comparison"
    prev=ANCESTRY_RUN/f"output/fold_{fold}"
    early=json.loads((RUN/f"output/fold_{fold}/early_stop.json").read_text())
    budget=early["completed_epochs"]
    epochs=[json.loads((prev/f"fit/epochs/epoch_{e:03d}.json").read_text()) for e in range(1,budget+1)]
    winner=max(epochs,key=lambda x:x["val_Significance20_30_50_epoch"])
    files=list((prev/"fit/checkpoints").glob(f"model-epoch=epoch={winner['epoch']-1}-*.ckpt"))
    if len(files)!=1:
        raise ValueError("Previous equal-budget checkpoint is missing or nonunique")
    cfg,model,_,name=read_checkpoint(prev,files[0])
    result=dict(fold=fold,max_epochs_considered=budget,previous_epoch=winner["epoch"],
                previous_selected_validation_z=winner["val_Significance20_30_50_epoch"],
                current_epoch=early["selected_epoch"],reference_checkpoint=name,reference_device="CPU",
                current_scores="standard GPU application",splits={})
    path=out/"same_epoch_budget.json"
    rows=[]
    with TemporaryDirectory(prefix="synapse-equal-epochs-") as tmp:
        tmp=Path(tmp)
        for split in ("test","validation"):
            scored=tmp/(split+".root")
            apply(model,cfg,fold,split,scored)
            metrics,entries=compare(fold,split,scored)
            result["splits"][split]={"same_split":metrics}
            rows.extend(entries)
            publish(path,result)
        val=result["splits"]["validation"]["same_split"]
        cuts={k:val[k]["cuts"] for k in ("previous_budget","current")}
        metrics,entries=compare(fold,"test",tmp/"test.root",cuts)
        result["splits"]["test"]["validation_cuts"]=metrics
        rows.extend(entries)
        publish(path,result)
        with (out/"same_epoch_budget_processes.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        # Publish only the native numerical summary. Its intermediate ROOT and HDF5 files are local scratch.
        benchmark=REPO/"configs/run3_common/run_hhml_benchmark.py"
        subprocess.run(["bash","-c",'source "$1" && shift && exec python3 -B -u "$@"',"bash",LCG,str(benchmark),
                        "--fold",str(fold),"--validation",str(tmp/"validation.root"),"--test",str(tmp/"test.root"),
                        "--out",str(tmp/"native")],check=True)
        shutil.copyfile(tmp/"native/summary.json",out/"same_epoch_budget_trafo60.json")

    from crossfold_review import render
    render()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold",type=int,choices=(0,),required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main(args.fold)
