"""Build row-aligned MC Higgs labels from the indexed EasyJet files."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import uproot
from paths import PHYSICS, PLAN, PRODUCTION, RECIPE, REPO, RUN

VERSION = 1
FIELDS = {"ids": "i1", "ancestry": "i1", "known": "i2", "same": "i2"}


def jobs_for(plan):
    return [job for job in plan["jobs"] if job["kind"] == "weights"
            and not plan["samples"][job["sample"]]["is_data"]]


def identity(numbers, pt):
    return np.rec.fromarrays([numbers.astype("u8"), *pt[:, :3].astype("f4").T],
                             names="event,l1,l2,l3")


def read_labels(job, numbers):
    import ROOT
    if not hasattr(ROOT, "higgsdecay"):
        for directory in (PRODUCTION/"source/include", REPO/"configs/run3_common", RECIPE):
            ROOT.gInterpreter.AddIncludePath(str(directory))
        if not ROOT.gInterpreter.Declare('#include "higgs_decay.hxx"'):
            raise RuntimeError("Cannot compile the Higgs daughter matcher")
    ROOT.synphysics.selectEvents(ROOT.VecOps.AsRVec(np.ascontiguousarray(np.unique(numbers),dtype="u8")))
    df = ROOT.RDataFrame("AnalysisMiniTree", job["files"]).Filter("synphysics::wanted.count(eventNumber)")
    absent = "truth_H1_pdgId==0 && truth_H2_pdgId==0"
    zero = df.Filter(absent).AsNumpy(["eventNumber"], lazy=True)
    with_h = df.Filter("!(" + absent + ")")
    any_h = with_h.AsNumpy(["eventNumber"], lazy=True)
    for obj, name, pdg in (("el", "Electron", 11), ("mu", "Muon", 13)):
        flag = obj + "_isAnalysis" + name + "_NOSYS"
        cols = [obj+"_"+v for v in ("pt_NOSYS","eta","phi","charge")]
        cols += [flag]*4 + [obj+"_truthType", obj+"_truthOrigin",
                             "el_firstEgMotherPdgId" if obj=="el" else "h3::zeros<int>(mu_eta.size())", str(pdg)]
        with_h = with_h.Define("dec_"+obj, "h3::leptons("+",".join(cols)+")")
    with_h = with_h.Define("lep","h3::order(dec_el,dec_mu)").Filter("lep.size()==3")
    jet = "recojet_antikt4PFlow_"
    cols = [jet+v for v in ("pt_NOSYS","eta","phi","pt_NOSYS","isAnalysisJet_NOSYS","isAnalysisJet_NOSYS")]
    with_h = with_h.Define("jets","h3::jets("+",".join(cols)+",lep,false,true)")
    cols = ["lep","jets",*[jet+v for v in ("pt_NOSYS","eta","phi","isAnalysisJet_NOSYS")]]
    for h in (1,2):
        cols += [f"truth_H{h}_pdgId"]
        for generation in ("children","grandchildren"):
            cols += [f"truth_{generation}_fromH{h}_"+v for v in ("pdgId","eta","phi")]
    with_h = with_h.Define("higgsTargets","higgsdecay::labels("+",".join(cols)+")")
    columns = ["eventNumber","truth_H1_pdgId","truth_H2_pdgId","searchOverflow"]
    with_h = with_h.Define("searchOverflow","higgsTargets.conservative")
    for i in range(3):
        key = f"ptSlot{i}"
        with_h = with_h.Define(key,f"float(lep[{i}].p4.Pt())")
        columns.append(key)
    for name in FIELDS:
        for i in range(12):
            key = name+str(i)
            with_h = with_h.Define(key,f"higgsTargets.{name}[{i}]")
            columns.append(key)
    detail = with_h.AsNumpy(columns, lazy=True)
    values = detail.GetValue()
    pt = np.column_stack([values[f"ptSlot{i}"] for i in range(3)])
    order = identity(values["eventNumber"],pt).argsort(kind="stable")
    keys = identity(values["eventNumber"][order],pt[order])
    data = {key:np.column_stack([values[key+str(i)] for i in range(12)])[order].astype(dtype)
            for key,dtype in FIELDS.items()}
    h1,h2 = values["truth_H1_pdgId"][order],values["truth_H2_pdgId"][order]
    state = (h1==25).astype("i1")+(h2==25).astype("i1")
    state[~np.isin(h1,(0,25)) | ~np.isin(h2,(0,25))] = -1
    return (zero.GetValue()["eventNumber"],any_h.GetValue()["eventNumber"],keys,data,state,
            int(values["searchOverflow"].sum()))


def build(job):
    out = RUN/"shards"/f"{job['id']:06}.npz"
    if out.is_file():
        with np.load(out) as f:
            if str(f["sample"])!=job["sample"] or int(f["version"])!=VERSION:
                raise RuntimeError(f"Incompatible existing shard {out}")
        return
    rows = {}
    source = PHYSICS/"shards"/out.with_suffix(".root").name
    with uproot.open(source,handler=uproot.source.file.MemmapSource) as f:
        for role in ("base","control"):
            if role in f:
                a = f[role].arrays(["eventNumber","pt"],library="np")
                rows[role] = a["eventNumber"], a["pt"][:,:3].copy()
    wanted = np.concatenate([n for n,pt in rows.values()]) if rows else np.array([],dtype="u8")
    result = dict(version=VERSION,sample=job["sample"],overflow=0,ambiguous_join=0)
    if len(wanted):
        zero, with_h, keys, data, state, result["overflow"] = read_labels(job,wanted)
        mixed = np.intersect1d(zero,with_h)
    for role,(n,pt) in rows.items():
        result.update({role+"_id":n,role+"_pt":pt})
        status = np.full(len(n),-1,dtype="i1")
        labels = {k:np.full((len(n),12),-1 if k in ("ids","ancestry") else 0,dtype=t) for k,t in FIELDS.items()}
        if len(n):
            is_zero = np.isin(n,zero)
            mark = np.isin(n,with_h)
            if not (is_zero|mark).all():
                raise ValueError(f"Missing selected raw event in {job['id']} {role}")
            status[is_zero & ~mark] = 0
            lookup = identity(n,pt)
            lo,hi = keys.searchsorted(lookup,"left"),keys.searchsorted(lookup,"right")
            if np.any(mark & ~is_zero & (lo==hi)):
                raise ValueError(f"Missing selected H event or pT mismatch in {job['id']} {role}")
            unique = mark & (hi-lo==1) & ~np.isin(n,mixed)
            status[unique] = state[lo[unique]]
            for k in labels:
                labels[k][unique] = data[k][lo[unique]]
            result["ambiguous_join"] += int((mark & ~unique).sum())
        result[role+"_state"] = status
        for key,values in labels.items():
            result[role+"_"+key] = values
    with NamedTemporaryFile(dir=out.parent,suffix=".npz",delete=False) as tmp:
        np.savez_compressed(tmp,**result)
    Path(tmp.name).replace(out)
    print(job["id"],job["sample"],len(wanted),"rows",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=int,nargs="+")
    p.add_argument("--group",type=int,default=0)
    p.add_argument("--groups",type=int,default=1)
    args=p.parse_args()
    if not 0<=args.group<args.groups:
        p.error("Invalid group")
    jobs=jobs_for(json.loads(PLAN.read_bytes()))
    if args.jobs:
        jobs=[j for j in jobs if j["id"] in args.jobs]
        if len(jobs)!=len(set(args.jobs)): p.error("Unknown job ID")
    else:
        jobs=jobs[args.group::args.groups]
    (RUN/"shards").mkdir(parents=True,exist_ok=True)
    for job in jobs:
        build(job)
