"""Check calibrated HH hypotheses on the previous model's fold 1 validation tail."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

from hypothesis import INPUTS, features


def study(reference, out):
    reference=reference.resolve()
    with (reference/"output/fold_1/evaluation/validation/background_rejection/working_points.csv").open() as f:
        cuts={r["method"]:float(r["score_min"]) for r in csv.DictReader(f)
              if r["normalization"]=="from_preselection" and r["category"]=="VVV" and float(r["target_signal_eff"])==.2}
    plan=json.loads((reference.parents[3]/"caf/hhml3l0tau/work/nominal-run3-20261122/plan/plan.json").read_bytes())
    catalog={int(s["dsid"]):s["description"] for s in plan["samples"].values() if s["process"]=="VVV"}
    store=defaultdict(list)
    kw=dict(handler=uproot.source.file.MemmapSource,array_cache=None)
    with uproot.open(reference/"dataset/3fold_2.root",**kw) as raw, uproot.open(reference/"output/fold_1/validation/validation_scores.root",**kw) as scored:
        a,b=raw["tree"],scored["tree"]
        if a.num_entries!=b.num_entries:
            raise ValueError("Misaligned validation file sizes")
        for start in range(0,a.num_entries,32000):
            opts=dict(entry_start=start,entry_stop=start+32000,library="np")
            s=b.arrays(["eventNumber","dsid","n_jet","class_id","weight","p_signal","p_extconv_bkg",
                        "evenet_pS","evenet_pConv","evenet_valid"],**opts)
            nn=(s["p_signal"]>=cuts["Synapse"]) & (s["p_extconv_bkg"]<.8)
            eve=(s["evenet_pS"]>=cuts["EveNet"]) & (s["evenet_pConv"]<.8) & s["evenet_valid"].astype(bool)
            take=(s["n_jet"]>=1) & ((s["class_id"]==0)|np.isin(s["dsid"],list(catalog))) & (nn|eve)
            if not take.any():
                continue
            d=a.arrays([*INPUTS,"eventNumber","dsid","isPrompt","wAssignment"],**opts)
            for key in ("dsid","eventNumber"):
                np.testing.assert_array_equal(d[key],s[key])
            data={k:v[take] for k,v in d.items()}
            observables=features(data)
            for key in ("dsid","weight","class_id","n_jet"):
                store[key].append(s[key][take])
            for key,value in dict(nn=nn[take],eve=eve[take],prompt=data["isPrompt"][:,:3].sum(1),**observables).items():
                store[key].append(value)
    d={k:np.concatenate(v) for k,v in store.items()}
    hh=d["class_id"]==0
    vz=~hh
    weight=d["weight"].astype("f8")
    wpos=np.maximum(0,weight)
    summary=[]
    groups=[("HH",hh),("VVV",vz)] + [(catalog[k],d["dsid"]==k) for k in sorted(catalog)]
    for group, mask in groups:
        nn=d["nn"] & mask
        ev=d["eve"] & mask
        has2=d["hh_two_jet_valid"].astype(bool)
        has1=d["hh_one_jet_valid"].astype(bool)
        den=wpos[nn].sum()
        def fraction(sub,base):
            total=wpos[base].sum()
            return float(wpos[base&sub].sum()/total) if total else None
        summary.append(dict(sample=group,previous_selected=int(nn.sum()),previous_signed_yield=float(weight[nn].sum()),
            evenet_signed_yield=float(weight[ev].sum()),paired_mc_error=float(np.sqrt((weight[mask&(nn!=ev)]**2).sum())),
            positive_weight=float(den),three_prompt_fraction=fraction(d["prompt"]==3,nn),two_jet_available_fraction=fraction(has2,nn),
            two_jet_below125_given_available=fraction(d["hh_two_jet_bound"]<=125,nn&has2),
            one_jet_below125_given_only_one=fraction(d["hh_one_jet_bound"]<=125,nn&has1&~has2),
            mean_fraction_compatible_two_jet=float(np.average(d["hh_two_jet_compatible_fraction"][nn&has2],
                weights=wpos[nn&has2])) if wpos[nn&has2].sum() else None))
    out.mkdir(parents=True,exist_ok=True)
    (out/"validation_study.json").write_text(json.dumps(dict(split="previous fold 1 validation", cuts=cuts,
        description="Selected at each original 20 percent HH threshold. Geometry rates use positive nominal weights within the indicated subset.",rows=summary),indent=2)+"\n")
    for r in summary:
        if r["sample"] in ("HH","VVV") or r["previous_signed_yield"]>.02:
            print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    study(args.reference,args.out)
