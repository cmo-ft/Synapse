"""Add calibrated HH visible hypotheses to each partition of the parent-presence dataset."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot

from hypothesis import FEATURES, INPUTS, features


def open_root(path):
    return uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None)


def scalar_reference(sample):
    """Independent scalar four-vector construction for one event, including incomplete topologies."""
    valid = sample["valid"].astype(bool)
    jet_slots = [j for j in range(3,9) if valid[j] and sample["isJet"][j]]
    v = {}
    for j in [0,1,2,*jet_slots]:
        pt=float(sample["pt"][j])*.001
        eta,phi=map(float,(sample["eta"][j],sample["phi"][j]))
        v[j]=np.array([float(sample["e"][j])*.001, pt*np.cos(phi),pt*np.sin(phi),pt*np.sinh(eta)])
    def m(slots):
        total=sum((v[i] for i in slots),start=np.zeros(4))
        return np.sqrt(max(0.,total[0]**2-np.dot(total[1:],total[1:])))
    pools={1:[],2:[]}
    for i,j in combinations(range(3),2):
        if not valid[:3].all() or sample["charge"][i]*sample["charge"][j]>=0:
            continue
        l=3-i-j
        mll=m((i,j))
        dp=float(sample["phi"][i])-float(sample["phi"][j])
        dr=np.hypot(float(sample["eta"][i])-float(sample["eta"][j]),np.arctan2(np.sin(dp),np.cos(dp)))
        for size in (1,2):
            for jets in combinations(jet_slots,size):
                mh=m((l,*jets))
                pools[size].append((max(mll,mh),mh,mll,dr,m(jets) if size==2 else 0.))
    one,two=[min(pools[size],key=lambda row:row[:2]) if pools[size] else (0.,)*5 for size in (1,2)]
    selected=two if pools[2] else one
    return (bool(pools[1]),one[0],bool(pools[2]),two[0],two[4],
            sum(x[0]<=125. for x in pools[2])/max(len(pools[2]),1),selected[1],selected[2],selected[3])


def hashes(keys):
    return {k:hashlib.sha256() for k in keys}


def update(checks, data):
    for k,v in checks.items():
        v.update(np.ascontiguousarray(data[k]).view("u1"))


def build(source,dest,source_audit):
    checks={}
    nrows=n1=n2=n_scalar=0
    with open_root(source) as root, uproot.recreate(dest,compression=uproot.LZ4(4)) as out:
        tree=root["tree"]
        original=tree.typenames()
        if set(FEATURES)&original.keys():
            raise ValueError("Source already contains derived hypotheses")
        checks=hashes([*original,*FEATURES])
        writer=None
        for a in tree.iterate(step_size=20000,library="np"):
            f=features(a)
            if n_scalar<96:
                for row in np.linspace(0,len(a["pt"])-1,min(12,len(a["pt"])),dtype=int):
                    scalar=scalar_reference({k:a[k][row] for k in INPUTS})
                    np.testing.assert_allclose([f[k][row] for k in FEATURES],scalar,rtol=2e-5,atol=3e-4)
                    n_scalar+=1
            if np.any(f[FEATURES[2]]>f[FEATURES[0]]) or np.any((f[FEATURES[5]]<0)|(f[FEATURES[5]]>1)):
                raise ValueError("Invalid hypothesis availability or candidate count")
            a.update(f)
            update(checks,a)
            if writer is None:
                writer=out.mktree("tree",{k:np.dtype((v.dtype,v.shape[1:])) for k,v in a.items()})
            writer.extend(a)
            nrows+=len(a["pt"])
            n1+=int(f[FEATURES[0]].sum())
            n2+=int(f[FEATURES[2]].sum())
        if nrows!=tree.num_entries:
            raise ValueError("Missing input rows")
    original_hash=source_audit["sha256_by_branch"]
    if original_hash.keys()!=original.keys() or any(checks[k].hexdigest()!=s for k,s in original_hash.items()):
        raise ValueError("Original branch bytes differ from the source publication")
    verified=hashes(checks.keys())
    with open_root(dest) as root:
        tree=root["tree"]
        if tree.num_entries!=nrows or any(tree.typenames()[k]!=v for k,v in original.items()):
            raise ValueError("Output row count or original branch schema changed")
        for a in tree.iterate(step_size=24000,library="np"):
            update(verified,a)
    if any(v.digest()!=verified[k].digest() for k,v in checks.items()):
        raise ValueError("Read-back verification failed")
    result=dict(rows=nrows,one_jet=n1,two_jet=n2,scalar_checks=n_scalar,sha256_by_branch={k:v.hexdigest() for k,v in verified.items()})
    print(source.name,{k:v for k,v in result.items() if k!="sha256_by_branch"},flush=True)
    return result


def main(run,reference):
    target=run/"dataset"
    if target.exists():
        audit=json.loads((target/"audit.json").read_bytes())
        if audit["reference"]!=reference.name or audit["features"]!=list(FEATURES) or len(audit["splits"])!=6:
            raise ValueError("Conflicting dataset at "+str(target))
        return
    prior=json.loads((reference/"dataset/audit.json").read_bytes())
    with TemporaryDirectory(dir=run,prefix=".physics-") as temp:
        new=Path(temp)/"dataset"
        new.mkdir()
        reports={}
        for rem in (0,2,1):
            for role,prefix in (("base","3fold"),("control","wzcr")):
                key=f"{role}_{rem}"
                name=f"{prefix}_{rem}.root"
                reports[key]=build(reference/"dataset"/name,new/name,prior["splits"][key])
        audit=dict(reference=reference.name,features=list(FEATURES),splits=reports)
        (new/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
        summary=json.loads((reference/"dataset/summary.json").read_text())
        summary["visible_hypotheses"]=list(FEATURES)
        (new/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        new.rename(target)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path)
    p.add_argument("--reference",required=True,type=Path)
    a=p.parse_args()
    main(a.run.resolve(),a.reference.resolve())
