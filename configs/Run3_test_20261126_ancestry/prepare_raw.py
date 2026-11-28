"""Recover positive raw masks and match explicit Higgs decay daughters for selected MC rows."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from paths import PLAN, RECIPE, REFERENCE, REPO, RUN


def jobs_for(plan):
    return [j for j in plan["jobs"] if j["kind"] == "weights"
            and (s := plan["samples"][j["sample"]]) and not s["is_data"]
            and (s["is_signal"] or s["process"] in ("singleH", "top"))]


def identity(ids, pt):
    return np.rec.fromarrays([ids.astype("u8"), *pt[:, :3].astype("f4").T],
                             names="event,l1,l2,l3")


def read_truth(job, sample, numbers):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT, "ancestry") or not hasattr(ROOT.ancestry, "labels"):
        for folder in (REPO / "configs/run3_common", RECIPE):
            ROOT.gInterpreter.AddIncludePath(str(folder))
        if not ROOT.gInterpreter.Declare('#include "ancestry.hxx"'):
            raise RuntimeError("Failed to compile ancestry matcher")
    ROOT.synphysics.selectEvents(ROOT.VecOps.AsRVec(np.ascontiguousarray(np.unique(numbers), dtype="u8")))
    df = ROOT.RDataFrame("AnalysisMiniTree", job["files"]).Filter("synphysics::wanted.count(eventNumber)")
    for obj, name, pdg in (("el", "Electron", 11), ("mu", "Muon", 13)):
        cols = [obj+"_"+s for s in ("pt_NOSYS","eta","phi","charge","isAnalysis"+name+"_NOSYS")]
        df = df.Define("ancestry_"+obj, "synphysics::selectedLeptons("+",".join(cols)+f",{pdg})")
    df = df.Define("ancestry_leptons","h3::order(ancestry_el,ancestry_mu)").Filter("ancestry_leptons.size()==3")
    jet = "recojet_antikt4PFlow_"
    cols = [jet+x for x in ("pt_NOSYS","eta","phi","pt_NOSYS","isAnalysisJet_NOSYS","isAnalysisJet_NOSYS")]
    df = df.Define("ancestry_jets","h3::jets("+",".join(cols)+",ancestry_leptons,false,true)")
    args = ["ancestry_leptons","ancestry_jets",*[jet+x for x in ("pt_NOSYS","eta","phi","isAnalysisJet_NOSYS")]]
    args += [prefix+"_parent"+p+"ParentsMask" for prefix in ("el","mu",jet[:-1]) for p in ("Higgs","Top")]
    available = set(sample["branches"])
    for h in (1,2):
        names = [f"truth_children_fromH{h}_pdgId",*[f"truth_grandchildren_fromH{h}_"+q for q in ("pdgId","eta","phi")]]
        if all(k in available for k in names):
            args.extend(names)
        else:
            args.extend(("std::vector<int>{}","std::vector<std::vector<int>>{}",
                         "std::vector<std::vector<float>>{}","std::vector<std::vector<float>>{}"))
    df = df.Define("ancestry_bits","ancestry::labels("+",".join(args)+")")
    columns=["eventNumber"]
    for i in range(3):
        name=f"slotPt{i}"
        df=df.Define(name,f"float(ancestry_leptons[{i}].p4.Pt())")
        columns.append(name)
    for i in range(12):
        name=f"bits{i}"
        df=df.Define(name,f"unsigned(ancestry_bits[{i}])")
        columns.append(name)
    return df.AsNumpy(columns)


def build(job, sample, dest):
    filename=f"{job['id']:06}.npz"
    out=dest/filename
    if out.exists():
        with np.load(out) as f:
            if str(f["sample"])!=job["sample"] or int(f["version"])!=1:
                raise RuntimeError(f"Unexpected existing shard {out}")
        return
    with np.load(REFERENCE/"parent_shards"/filename) as old:
        rows={role:(old[role+"_id"],old[role+"_pt"]) for role in ("base","control") if role+"_id" in old}
    arrays={"sample":job["sample"],"version":1}
    numbers=np.concatenate([ids for ids,_ in rows.values()]) if rows else np.array([],dtype="u8")
    if len(numbers):
        raw=read_truth(job,sample,numbers)
        pt=np.column_stack([raw[f"slotPt{i}"] for i in range(3)])
        keys=identity(raw["eventNumber"],pt)
        order=keys.argsort(kind="stable")
        keys=keys[order]
        bits=np.column_stack([raw[f"bits{i}"] for i in range(12)]).astype("u1")
    for role,(ids,pt) in rows.items():
        result=np.zeros((len(ids),12),"u1")
        if len(ids):
            wanted=identity(ids,pt)
            left,right=np.searchsorted(keys,wanted,"left"),np.searchsorted(keys,wanted,"right")
            if (left==right).any():
                raise ValueError(f"Missing source rows for {filename}/{role}")
            unique=(right-left)==1
            result[unique]=bits[order[left[unique]]]
        arrays.update({role+"_id":ids,role+"_pt":pt,role+"_bits":result})
    with NamedTemporaryFile(dir=dest,suffix=".npz",delete=False) as f:
        np.savez_compressed(f,**arrays)
    Path(f.name).replace(out)
    print(job["id"],job["sample"],len(numbers),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group",type=int,default=0)
    parser.add_argument("--groups",type=int,default=1)
    parser.add_argument("--job",type=int)
    args=parser.parse_args()
    plan=json.loads(PLAN.read_bytes())
    jobs=jobs_for(plan)
    if args.job is not None:
        jobs=[j for j in jobs if j["id"]==args.job]
        if not jobs: parser.error("Unknown job")
    else:
        if not 0<=args.group<args.groups: parser.error("Invalid group")
        jobs=jobs[args.group::args.groups]
    dest=RUN/"raw_labels"
    dest.mkdir(parents=True,exist_ok=True)
    for job in jobs: build(job,plan["samples"][job["sample"]],dest)
