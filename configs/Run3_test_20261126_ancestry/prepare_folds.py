"""Publish checked, row-aligned training folds with new ancestry and W assignment labels."""
import argparse
import time
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import uproot
from paths import PARENTS, PLAN, REFERENCE, RUN
from prepare_raw import jobs_for

WZ = (701005, 701045)
TTBAR = (601229, 601230, 601237)
TTH = (602637, 602638, 604224)
GROUPS = {"VV__WZ":"WZ", "VV__otherVV":"OtherVV"}


def add_labels(a, bits, catalog):
    valid = a["valid"].astype(bool) & ~a["isMET"].astype(bool)
    dsid = a["dsid"]
    hh = a["label_signal"].astype(bool)
    shape = valid.shape
    ancestry = {p: np.full(shape, -1, np.int8) for p in PARENTS}
    h,t,w,z = (ancestry[p] for p in PARENTS)
    rw, rz = a["relationW"], a["relationZ"]
    h[(a["relationHiggs"]>0) | ((bits&1)!=0)] = 1
    t[(a["relationTop"]>0) | ((bits&2)!=0)] = 1
    from_hw = ((bits&4)!=0) | (rw>0)
    # An explicit H -> ZZ assignment uses the generator in the Higgs-bearing samples.
    has_h = hh | np.isin(dsid, catalog["higgs"])
    from_hz = ((bits&8)!=0) | ((rz>0) & has_h[:,None] & ~np.isin(dsid, (*WZ,701040,701105))[:,None])
    h[from_hw | from_hz] = 1
    w[from_hw] = 1
    z[(rz>0) | from_hz] = 1
    z[rz==0] = 0
    # Exclusive prompt boson daughters identify the other boson species as a nonmember.
    # Assign competing species only when the hard-decay matches are unambiguous.
    w[from_hz & ~from_hw] = 0
    z[from_hw & ~from_hz] = 0
    t[(from_hw | from_hz) & hh[:,None]] = 0
    wz = np.isin(dsid,WZ)
    known_wz = (rz[:,:3]>=0).all(1) & ((rz[:,:3]==0).sum(1)==1) & a["isPrompt"][:,:3].all(1)
    continuum = wz & known_wz
    w[continuum,:3] = (rz[continuum,:3]==0).astype("i1")
    h[continuum,:3] = t[continuum,:3] = 0
    zz = (dsid==701040)[:,None] & (rz>0)
    w[zz] = h[zz] = t[zz] = 0
    # A top-matched object in exclusive ttbar has independent evidence against Higgs ancestry.
    h[np.isin(dsid,TTBAR)[:,None] & (t==1)] = 0
    origin = a["partOrigin"]
    lep = np.isin(np.abs(a["pdg"]), (11,13)) & a["isPrompt"].astype(bool)
    tth_top = np.isin(dsid,TTH)[:,None] & (t==1) & lep & (origin==2)
    associated_w = np.isin(dsid,(601501,601502))[:,None] & lep & (origin==4)
    h[(tth_top | associated_w) & ~from_hw & ~from_hz] = 0
    w[associated_w & ~from_hz] = 1
    for p, labels in ancestry.items():
        labels[~valid] = -1
        a["ancestry"+p] = labels

    # Only the fully resolved HH -> 4W topology certifies all alternative central
    # jet pairs as wrong assignments. Preserve generic per-object W ancestry above.
    j = rw[:,3:9]
    known_jet = valid[:,3:9] & (j>0)
    i,k = np.triu_indices(6,1)
    pair = known_jet[:,i] & known_jet[:,k] & (j[:,i]==j[:,k])
    leps = rw[:,:3]
    full = hh & ((leps>0) & valid[:,:3]).all(1) & (known_jet.sum(1)==2) & (pair.sum(1)==1)
    full &= (leps[:,0]!=leps[:,1]) & (leps[:,0]!=leps[:,2]) & (leps[:,1]!=leps[:,2])
    daughters = np.max(np.where(known_jet,j,-1),axis=1)
    full &= (leps!=daughters[:,None]).all(axis=1)
    new_w = rw.copy()
    other = full[:,None] & valid[:,3:9] & ~known_jet
    new_w[:,3:9][other] = 0
    a["relationW"] = new_w
    a["wAssignment"] = full
    a["isWZ"] = np.isin(dsid, catalog["wz"])
    return full


def indexed_rows(jobs, role, rem):
    rows = defaultdict(list)
    for job in jobs:
        with np.load(RUN/"raw_labels"/f"{job['id']:06}.npz") as f:
            if str(f["sample"])!=job["sample"] or int(f["version"])!=1:
                raise ValueError(f"Invalid ancestry shard {job['id']}")
            if role+"_id" not in f: continue
            ids = f[role+"_id"]
            keep = ids%3==rem
            if keep.any():
                rows[int(job["sample"].split("_")[0])].append((ids[keep],f[role+"_pt"][keep],f[role+"_bits"][keep]))
    return {d:tuple(np.concatenate(t) for t in zip(*parts)) for d,parts in rows.items()}


def build(label, rem, jobs, catalog, stage):
    role = "base" if label=="3fold" else "control"
    table = indexed_rows(jobs,role,rem)
    offsets, counts = Counter(), Counter()
    name=f"{label}_{rem}.root"
    with uproot.open(REFERENCE/"dataset"/name,handler=uproot.source.file.MemmapSource) as src, uproot.recreate(stage/name,compression=uproot.LZ4(4)) as out:
        t=src["tree"]
        writer=None
        for a in t.iterate(step_size=24000,library="np"):
            if np.any(a["eventNumber"]%3!=rem) or np.any(a["in_wzcr"]!=(role=="control")):
                raise ValueError(f"Incorrect role or event remainder in {name}")
            dsids=a["dsid"]
            bits=np.zeros_like(a["valid"],dtype=np.uint8)
            for d in np.unique(dsids):
                if d not in table: continue
                idx=dsids==d
                n=int(idx.sum())
                sl=slice(offsets[d],offsets[d]+n)
                ids,pt,b=table[d]
                np.testing.assert_array_equal(ids[sl],a["eventNumber"][idx])
                np.testing.assert_array_equal(pt[sl],a["pt"][idx,:3])
                bits[idx]=b[sl]
                offsets[d]+=n
            full=add_labels(a,bits,catalog)
            counts["events"]+=len(dsids)
            for cls in range(3):
                counts[f"class_abs_sow/{cls}"]+=np.abs(a["weight"][a["class_id"]==cls]).sum(dtype="f8")
            counts["W_fully_matched"]+=int(full.sum())
            njet=a["valid"][full,3:9].sum(1).astype(int)
            counts["W_alternative_central_pairs"]+=int((njet*(njet-1)//2-1).sum())
            for d in np.unique(dsids):
                group=catalog["process"][int(d)]
                event=dsids==d
                for name_p in PARENTS:
                    y=a["ancestry"+name_p][event]
                    for kind,sl in (("lepton",slice(0,3)),("central",slice(3,9)),("forward",slice(10,12))):
                        vals=y[:,sl][a["valid"][event,sl].astype(bool)]
                        hist=np.bincount(vals.astype(int)+1,minlength=3)
                        for tag,n in zip(("unknown","negative","positive"),hist):
                            counts[f"{group}/{name_p}/{kind}/{tag}"]+=int(n)
            if writer is None:
                writer=out.mktree("tree",{k:np.dtype((v.dtype,v.shape[1:])) for k,v in a.items()})
            writer.extend(a)
        if counts["events"]!=t.num_entries or any(offsets[d]!=len(v[0]) for d,v in table.items()):
            raise RuntimeError("Incomplete row coverage for "+name)
    print(name, counts["events"],counts["W_fully_matched"],counts["W_alternative_central_pairs"],flush=True)
    return dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true",help="Wait for submitted raw readers before publishing")
    wait=parser.parse_args().wait
    plan=json.loads(PLAN.read_bytes())
    samples=[s for s in plan["samples"].values() if not s["is_data"]]
    catalog={"higgs":sorted({int(s["dsid"]) for s in samples if s["process"]=="singleH" or int(s["dsid"])==525360}),
             "wz":sorted({int(s["dsid"]) for s in samples if s["process"]=="VV__WZ"}),
             "process":{int(s["dsid"]):("HH" if s["is_signal"] else GROUPS.get(s["process"],s["process"])) for s in samples}}
    jobs=sorted(jobs_for(plan), key=lambda j:(j["sample"],j["id"]))
    absent=[j["id"] for j in jobs if not (RUN/"raw_labels"/f"{j['id']:06}.npz").is_file()]
    while absent:
        if not wait: raise SystemExit(f"Missing {len(absent)} source shards: {absent[:15]}")
        print(f"Waiting for {len(absent)} raw shards",flush=True)
        time.sleep(20)
        absent=[i for i in absent if not (RUN/"raw_labels"/f"{i:06}.npz").is_file()]
    dest=RUN/"dataset"
    if (dest/"audit.json").exists(): return
    with TemporaryDirectory(dir=RUN,prefix=".publish-") as tmp:
        stage=Path(tmp)/"dataset"
        stage.mkdir()
        counts={f"{label}_{i}":build(label,i,jobs,catalog,stage) for i in (2,0,1) for label in ("3fold","wzcr")}
        (stage/"audit.json").write_text(json.dumps(counts,indent=2)+"\n")
        summary=json.loads((REFERENCE/"dataset/summary.json").read_bytes())
        summary["sum_abs_weights_by_remainder"]=[[counts[f"3fold_{i}"][f"class_abs_sow/{cls}"]
                                                 for cls in range(3)] for i in range(3)]
        summary["ancestry_label_version"]=1
        (stage/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        stage.rename(dest)


if __name__=="__main__":
    main()
