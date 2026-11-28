"""Join Higgs labels to the frozen folds and audit old and new targets before publication."""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot

from paths import BASELINE, PLAN, RUN
from prepare_raw import FIELDS, VERSION, jobs_for

N = 12
I,J = np.triu_indices(N,1)
BITS = (1 << J).astype("i2")
LEP = np.arange(N)<3


def pack(pairs):
    packed=np.zeros((len(pairs),N),dtype="i2")
    for i in range(N-1):
        cols=I==i
        packed[:,i]=(pairs[:,cols]*BITS[cols]).sum(1)
    return packed


def matrix(ids):
    left,right=ids[:,I],ids[:,J]
    return np.where((left>=0)&(right>=0), ((left>0)&(left==right)).astype("i1"), -1)


def shard_table(jobs,role,rem):
    parts=defaultdict(list)
    for job in jobs:
        path=RUN/"shards"/f"{job['id']:06}.npz"
        with np.load(path) as f:
            if int(f["version"])!=VERSION or str(f["sample"])!=job["sample"]:
                raise ValueError(f"Wrong source shard {path}")
            if role+"_id" not in f: continue
            n=f[role+"_id"]
            use=n%3==rem
            if not use.any(): continue
            data={name:f[role+"_"+name][use] for name in ("pt","state",*FIELDS)}
            data["id"]=n[use]
            parts[int(job["sample"].split("_")[0])].append(data)
    table={}
    for dsid in list(parts):
        rows=parts.pop(dsid)
        table[dsid]={k:np.concatenate([a[k] for a in rows]) for k in rows[0]}
    return table


def aligned(a,table,offsets):
    dsids=a["dsid"]
    out={k:np.empty((len(dsids),N),dtype=t) for k,t in FIELDS.items()}
    out["state"]=np.empty(len(dsids),dtype="i1")
    for d in np.unique(dsids):
        select=dsids==d
        count=int(select.sum())
        batch=slice(offsets[d],offsets[d]+count)
        source=table[d]
        np.testing.assert_array_equal(source["id"][batch],a["eventNumber"][select])
        np.testing.assert_array_equal(source["pt"][batch],a["pt"][select,:3])
        for k in out: out[k][select]=source[k][batch]
        offsets[d]+=count
    return out


def targets(a,raw,audit):
    valid=a["valid"].astype(bool) & ~a["isMET"].astype(bool)
    eligible=valid[:,I] & valid[:,J]
    ids,anc=raw["ids"],raw["ancestry"]
    empty=raw["state"]==0
    ids[empty]=0
    anc[empty]=0
    ids[~valid]=anc[~valid]=-1
    y=np.where((raw["known"][:,I]&BITS)!=0,((raw["same"][:,I]&BITS)!=0).astype("i1"),-1)
    full=a["wAssignment"].astype(bool)
    rw=a["relationW"]
    anchor=full[:,None] & (rw>0) & valid
    if np.any(rw[anchor]>4) or np.any(empty & full):
        raise ValueError("Inconsistent Higgs record for a fully matched four-W event")
    expected=(rw+1)//2
    if np.any(anchor & (ids>=0) & (expected!=ids)):
        raise ValueError("Unique raw decay assignment disagrees with the frozen five-object W match")
    audit["additional_anchored_objects"]+=int((anchor & (ids<0)).sum())
    ids[anchor]=expected[anchor]
    anc[anchor]=1
    left,right=ids[:,I],ids[:,J]
    forced=(left==0)|(right==0)|((left>0)&(right>0))
    updated=((left>0)&(left==right)).astype("i1")
    if ((y>=0)&forced&(updated!=y)&eligible).any():
        raise ValueError("Inconsistent individual and joint Higgs assignments")
    y[forced]=updated[forced]
    y[~eligible]=-1
    if np.any((ids>0)&(anc!=1)) or np.any((ids==0)&(anc!=0)):
        raise ValueError("Inconsistent object/instance labels")
    if full.any():
        pair_anchor=anchor[full][:,I] & anchor[full][:,J]
        truth=y[full]
        if not ((pair_anchor.sum(1)==10).all() and ((pair_anchor & (truth==1)).sum(1)==4).all()
                and ((pair_anchor & (truth==0)).sum(1)==6).all()):
            raise ValueError("Resolved HH requires four same-Higgs and six different-Higgs pairs")
    return ids,anc,y,valid,eligible,anchor


def confusion(old,new,mask):
    code=(old[mask].astype("i2")+1)*3+new[mask]+1
    return np.bincount(code,minlength=9).reshape(3,3)


def build(rem,role,jobs,process,stage):
    label="3fold" if role=="base" else "wzcr"
    name=f"{label}_{rem}.root"
    table=shard_table(jobs,role,rem)
    offsets,counts=Counter(),Counter()
    changes=defaultdict(lambda:defaultdict(lambda:np.zeros((3,3),dtype="i8")))
    full_counts=defaultdict(lambda:np.zeros((2,3),dtype="i8"))
    with uproot.open(BASELINE/"dataset"/name,handler=uproot.source.file.MemmapSource) as root, \
            uproot.recreate(stage/name,compression=uproot.LZ4(4)) as dest:
        tree=root["tree"]
        writer=None
        for a in tree.iterate(step_size=20000,library="np"):
            if np.any(a["eventNumber"]%3!=rem) or np.any(a["in_wzcr"]!=(role=="control")):
                raise ValueError(f"Wrong split or region in {name}")
            raw=aligned(a,table,offsets)
            old=matrix(a["relationHiggs"])
            ids,anc,y,valid,eligible,anchor=targets(a,raw,counts)
            full=a["wAssignment"].astype(bool)
            counts["events"]+=len(full)
            counts["fully_matched_HH"]+=int(full.sum())
            for state,n in zip(*np.unique(raw["state"],return_counts=True)):
                counts[f"record_H_count/{state}"]+=int(n)
            jj=(I>=3)&(J>=3)
            ll=(I<3)&(J<3)
            categories=(("ll",ll),("lj",~(ll|jj)),("jj",jj))
            for d in np.unique(a["dsid"]):
                select=a["dsid"]==d
                rows=changes[process[int(d)]]
                for group,mask in (("lepton",LEP),("jet",~LEP)):
                    rows[group]+=confusion(a["ancestryHiggs"][select],anc[select],valid[select]&mask)
                for group,mask in categories:
                    rows[group]+=confusion(old[select],y[select],eligible[select]&mask)
            if full.any():
                old_f=old[full]
                anchorpair=anchor[full][:,I]&anchor[full][:,J]
                for group,mask in categories:
                    for target in (0,1):
                        take=anchorpair & mask & (y[full]==target)
                        full_counts[group][target]+=np.bincount(old_f[take]+1,minlength=3)
            for event,ds,expected in ((104982,604305,(1,0,1)),(8057976,604422,(2,2,1,1,1))):
                use=(a["eventNumber"]==event)&(a["dsid"]==ds)
                if use.any():
                    np.testing.assert_equal(ids[use,:len(expected)],np.broadcast_to(expected,(int(use.sum()),len(expected))))
                    counts[f"reference_event/{ds}/{event}"]+=int(use.sum())
            a.update(relationHiggs=ids,ancestryHiggs=anc,higgsPairKnown=pack(y>=0),higgsPairSame=pack(y==1))
            if writer is None: writer=dest.mktree("tree",{k:np.dtype((v.dtype,v.shape[1:])) for k,v in a.items()})
            writer.extend(a)
        if counts["events"]!=tree.num_entries or any(offsets[d]!=len(row["id"]) for d,row in table.items()):
            raise ValueError(f"Row coverage mismatch in {name}")
    print(name, dict(counts),flush=True)
    return dict(counts=dict(counts),process_changes={p:{k:v.tolist() for k,v in groups.items()} for p,groups in changes.items()},
                fully_matched_HH_old_labels_by_new={k:v.tolist() for k,v in full_counts.items()})


def main(wait=False):
    dest=RUN/"dataset"
    if (dest/"audit.json").is_file():
        print("Published",dest)
        return
    plan=json.loads(PLAN.read_bytes())
    jobs=sorted(jobs_for(plan),key=lambda j:(j["sample"],j["id"]))
    required={f"{j['id']:06}.npz" for j in jobs}
    while True:
        missing=required.difference(p.name for p in (RUN/"shards").iterdir())
        if not missing: break
        message=f"{len(missing)} of {len(required)} shards pending"
        if not wait: raise SystemExit(message)
        print(message,flush=True)
        time.sleep(30)
    process={int(s["dsid"]):("HH" if s["is_signal"] else s["process"]) for s in plan["samples"].values() if not s["is_data"]}
    with TemporaryDirectory(dir=RUN,prefix=".publish-") as tmp:
        stage=Path(tmp)/"dataset"
        stage.mkdir()
        splits={}
        for rem in (0,2,1):
            for role in ("base","control"):
                splits[f"{role}_{rem}"]=build(rem,role,jobs,process,stage)
        summary=json.loads((BASELINE/"dataset/summary.json").read_bytes())
        summary["higgs_decay_label_version"]=VERSION
        (stage/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        audit=dict(label_order=["unknown","negative","positive"],change_axes=["previous","new"],
                   fully_matched_axes=["new negative/positive","previous unknown/negative/positive"],splits=splits)
        (stage/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
        stage.rename(dest)
    print("Published",dest,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait",action="store_true")
    main(parser.parse_args().wait)
