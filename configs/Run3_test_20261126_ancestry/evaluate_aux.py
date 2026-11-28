"""Weighted object ROC AUC and hadronic-W assignment efficiency on an applied split."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import uproot


def main(scores,dest):
    parents=("Higgs","Top","W","Z")
    hist=defaultdict(lambda:np.zeros((2,4096),dtype="f8"))
    counts=defaultdict(lambda:np.zeros(2,dtype="i8"))
    win=total=0.
    n=0
    def record(name,p,y,use,w):
        use=use & (y>=0)
        if not use.any(): return
        score=p[use]
        if not (np.isfinite(score).all() and (score>=0).all() and (score<=1).all()):
            raise ValueError(f"Invalid score in {name}")
        truth=y[use].astype(int)
        bins=np.minimum((score*4096).astype(int),4095)
        flat=truth*4096+bins
        hist[name][:]+=np.bincount(flat,weights=np.broadcast_to(w[:,None],y.shape)[use],minlength=8192).reshape(2,4096)
        counts[name][:]+=np.bincount(truth,minlength=2)
    with uproot.open(scores,handler=uproot.source.file.MemmapSource) as f:
        fields=["weight","valid","isMET","isLepton","partOrigin","isConv","isHF","wAssignment","isWZ","p_recoil_wz"]
        fields += [f"p_lepton_{n}" for n in ("higgs","top","wzvv","conversion","hf")]
        for parent in parents:
            fields.extend(["ancestry"+parent,"relation"+parent,"p_ancestry_"+parent.lower(),"p_pair_"+parent.lower()])
        packed = "higgsPairKnown" in f["tree"]
        if packed:
            fields.extend(["higgsPairKnown", "higgsPairSame"])
        for a in f["tree"].iterate(fields,step_size=16000,library="np"):
            valid=a["valid"].astype(bool)
            use=valid & ~a["isMET"].astype(bool)
            lep=valid & a["isLepton"].astype(bool)
            wt=np.abs(a["weight"].astype("f8"))
            recoil_truth = a["isWZ"][:,None]
            record("wz_recoil",a["p_recoil_wz"][:,None],recoil_truth,np.ones_like(recoil_truth,bool),wt)
            o=a["partOrigin"]
            for key,y in (("higgs",o==1),("top",o==2),("wzvv",(o>=3)&(o<=5)),("conversion",a["isConv"]==1),("hf",a["isHF"]==1)):
                record("lepton_"+key,a["p_lepton_"+key],y,lep,wt)
            i,j=np.triu_indices(valid.shape[1],1)
            for parent in parents:
                k=parent.lower()
                record("ancestry_"+k,a["p_ancestry_"+k],a["ancestry"+parent],use,wt)
                ids=a["relation"+parent]
                p=a["p_pair_"+k]
                known=use[:,i]&use[:,j]&(ids[:,i]>=0)&(ids[:,j]>=0)
                same=(ids[:,i]>0)&(ids[:,i]==ids[:,j])
                if parent=="Higgs" and packed:
                    known=use[:,i]&use[:,j]&(((a["higgsPairKnown"][:,i]>>j)&1)!=0)
                    same=((a["higgsPairSame"][:,i]>>j)&1)!=0
                record("pair_"+k,p,same,known,wt)
                if parent=="W":
                    central=(i>=3)&(j<9)
                    eligible=known&central
                    events=a["wAssignment"].astype(bool)&(eligible&same).any(1)&(eligible&~same).any(1)
                    winner=np.where(eligible,p,-np.inf).argmax(1)
                    correct=same[np.arange(len(p)),winner]
                    win+=wt[events&correct].sum()
                    total+=wt[events].sum()
                    n+=int(events.sum())
    metrics={}
    for name,(neg,pos) in hist.items():
        auc=None if pos.sum()*neg.sum()==0 else float((pos*(np.cumsum(neg)-0.5*neg)).sum()/(pos.sum()*neg.sum()))
        metrics[name]=dict(auc_4096_bins=auc,negative=int(counts[name][0]),positive=int(counts[name][1]))
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(weight="absolute nominal MC weight",tasks=metrics,
        w_best_central_jet_pair=dict(eligible_events=n,weighted_accuracy=win/total if total else None)),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("scores",type=Path); p.add_argument("destination",type=Path)
    a=p.parse_args()
    main(a.scores,a.destination)
