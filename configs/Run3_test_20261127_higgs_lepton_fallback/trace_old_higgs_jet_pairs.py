"""Trace the old positive same-Higgs jet pairs in SR + ExtConvCR to their recorded Higgs daughters."""
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from count_higgs_jet_pairs import LEFT, RIGHT
from paths import MATRIX_RUN, PHYSICS_RUN, PLAN, RUN, TREE_RUN


# Complete set in the published old folds, found from the valid positive jet-pair IDs.
PAIR_COUNTS = {604426:1078, 546231:53, 525360:72, 602637:6}
JET = "recojet_antikt4PFlow"
OPTIONS = dict(handler=uproot.source.file.MemmapSource, array_cache=None)


def shard_events(job):
    filename = f"{job['id']:06}"
    with np.load(MATRIX_RUN/"parent_shards"/(filename+".npz")) as z, np.load(TREE_RUN/"shards"/(filename+".npz")) as tree:
        if "base_id" not in z:
            return {}
        number, pt, ids = z["base_id"], z["base_pt"], z["base_ids"][:,0,:]
        same = (ids[:, LEFT] > 0) & (ids[:, LEFT] == ids[:, RIGHT])
        indices = np.flatnonzero(same.any(1))
        if not len(indices):
            return {}
        np.testing.assert_array_equal(tree["base_id"][indices], number[indices])
        tk, ts = tree["base_known"][indices], tree["base_same"][indices]
        state = tree["base_state"][indices]
    with uproot.open(PHYSICS_RUN/"shards"/(filename+".root"), **OPTIONS) as source:
        data = source["base"].arrays(["eventNumber","pt","eta","phi","pdg","valid","isMET"], library="np")
    np.testing.assert_array_equal(data["eventNumber"][indices], number[indices])
    np.testing.assert_array_equal(data["pt"][indices,:3], pt[indices])
    result = defaultdict(list)
    for off, row in enumerate(indices):
        reco = {k:data[k][row] for k in ("pt", "eta", "phi", "pdg")}
        pairs = []
        for i,j in zip(LEFT[same[row]], RIGHT[same[row]]):
            if not (data["valid"][row,i] and data["valid"][row,j]) or data["isMET"][row,i] or data["isMET"][row,j]:
                continue
            truth = "unknown"
            if state[off]==0 or (int(tk[off,i]) >> int(j)) & 1:
                truth = "same" if (int(ts[off,i]) >> int(j)) & 1 else "negative"
            pairs.append((int(i), int(j), int(ids[row,i]), truth))
        if pairs:
            result[int(number[row])].append((reco, pairs))
    return result


def branches():
    fields = ["eventNumber"]
    for species in ("el","mu",JET):
        fields.extend(species+"_"+k for k in ("pt_NOSYS","eta","phi"))
    fields.append(JET+"_parentHiggsParentsMask")
    for h in (1,2):
        fields.append(f"truth_H{h}_pdgId")
        fields.extend(f"truth_{level}_fromH{h}_{v}"
                      for level in ("children","grandchildren") for v in ("pdgId","eta","phi"))
    return fields


def raw_slot(record, reco, slot):
    species = JET if slot>=3 else ("el" if abs(reco["pdg"][slot])==11 else "mu")
    pt, eta, phi = [np.asarray(record[species+"_"+k]) for k in ("pt_NOSYS", "eta", "phi")]
    angle = (phi - reco["phi"][slot] + np.pi) % (2*np.pi) - np.pi
    same_axis = (abs(eta-reco["eta"][slot])<1e-4) & (abs(angle)<1e-4)
    # The selected-jet pT can differ from the original jet pT. Require a unique
    # matching direction, and check the original per-jet mask separately.
    if slot < 3:
        same_axis &= np.isclose(pt, reco["pt"][slot], rtol=2e-5, atol=0)
    return np.flatnonzero(same_axis)


def distance(reco, slot, particle):
    pdg, eta, phi = particle
    return math.hypot(reco["eta"][slot]-eta, math.remainder(reco["phi"][slot]-phi,2*math.pi))


def both_jets(reco, pair, daughters):
    i,j = pair
    a,b = daughters
    return ((distance(reco,i,a)<.2 and distance(reco,j,b)<.2)
            or (distance(reco,i,b)<.2 and distance(reco,j,a)<.2))


def trace(record, reco, pair):
    modes = []
    out = dict(has_bb=False, matches_bb=False, has_cc=False, matches_cc=False,
               has_WW=False, has_WW_lnuqq=False,
               has_WW_taunuqq=False, matches_WW_lnuqq_jj=False, matches_WW_lnuqq_ljj=False,
               matches_WW_taunuqq_jj=False, incomplete_WW_record=False)
    for h in (1,2):
        pdg = int(record[f"truth_H{h}_pdgId"])
        if not pdg:
            continue
        if pdg!=25:
            raise ValueError(f"Invalid Higgs record {pdg}")
        def particles(level):
            return [record[f"truth_{level}_fromH{h}_{field}"] for field in ("pdgId","eta","phi")]
        child = list(zip(*particles("children"),strict=True))
        signs = [p[0] for p in child]
        modes.append(",".join(map(str,sorted(signs))))
        for flavor, name in ((5,"bb"), (4,"cc")):
            quarks = [(a,b) for a,b in combinations(child,2) if {a[0],b[0]}=={-flavor,flavor}]
            out["has_"+name] |= bool(quarks)
            out["matches_"+name] |= any(both_jets(reco,pair,qq) for qq in quarks)
        if 24 not in signs or -24 not in signs:
            continue
        out["has_WW"] = True
        gp,ge,gf = particles("grandchildren")
        ws = [j for j,s in enumerate(signs) if abs(s)==24]
        if len(ws)!=2 or any(j>=min(len(gp),len(ge),len(gf)) for j in ws):
            out["incomplete_WW_record"] = True
            continue
        decays = [list(zip(gp[j],ge[j],gf[j],strict=True)) for j in ws]
        if any(len(d)!=2 for d in decays):
            out["incomplete_WW_record"] = True
        for had,lep in (decays,decays[::-1]):
            if len(had)!=2 or len(lep)!=2 or not all(1<=abs(p[0])<=5 for p in had):
                continue
            leptons = [p for p in lep if abs(p[0]) in (11,13,15)
                       and -(abs(p[0])+1)*int(np.sign(p[0])) in [n[0] for n in lep]]
            if len(leptons)!=1:
                continue
            lepton = leptons[0]
            jets_match = both_jets(reco,pair,had)
            if abs(lepton[0])==15:
                out["has_WW_taunuqq"] = True
                out["matches_WW_taunuqq_jj"] |= jets_match
            else:
                out["has_WW_lnuqq"] = True
                out["matches_WW_lnuqq_jj"] |= jets_match
                out["matches_WW_lnuqq_ljj"] |= jets_match and any(
                    reco["pdg"][k]==lepton[0] and distance(reco,k,lepton)<.1 for k in range(3))
    return {"H_children_pdgIds": " / ".join(sorted(modes)), **out}


def audit_job(job, sample):
    events = shard_events(job)
    if not events:
        return []
    wanted = np.fromiter(events,dtype="u8")
    covered = set()
    rows = []
    for path in job["files"]:
        with uproot.open(path, **OPTIONS) as source:
            t = source["AnalysisMiniTree"]
            number = t["eventNumber"].array(library="np")
            hits = np.flatnonzero(np.isin(number, wanted))
            for block in np.unique(hits//16000):
                ix = hits[hits//16000==block]
                start,end = int(ix[0]),int(ix[-1]+1)
                recs = ak.to_list(t.arrays(branches(),entry_start=start,entry_stop=end)[ix-start])
                for rec in recs:
                    n = int(rec["eventNumber"])
                    for idx,(reco,pairs) in enumerate(events[n]):
                        if any(len(raw_slot(rec,reco,k))!=1 for k in range(3)):
                            continue
                        key = n,idx
                        if key in covered:
                            raise ValueError(f"Duplicate source row in {job['sample']} event {n}")
                        covered.add(key)
                        for i,j,h,truth in pairs:
                            for k in (i,j):
                                match = raw_slot(rec,reco,k)
                                if len(match)!=1 or rec[JET+"_parentHiggsParentsMask"][match[0]] != 1<<(h-1):
                                    raise ValueError(f"Source jet or positive raw mask differs in {job['sample']}/{n}")
                            rows.append(dict(dsid=sample["dsid"],sample=job["sample"],event=n,slot_i=i,slot_j=j,
                                             tree_higgs_pair=truth, **trace(rec,reco,(i,j))))
    if len(covered)!=sum(map(len,events.values())):
        raise ValueError(f"Incomplete source coverage for {job['sample']}")
    return rows


def main():
    plan = json.loads(PLAN.read_bytes())
    rows = []
    for job in plan["jobs"]:
        sample = plan["samples"][job["sample"]]
        if job["kind"]=="weights" and not sample["is_data"] and int(sample["dsid"]) in PAIR_COUNTS:
            new = audit_job(job,sample)
            print(job["sample"],len(new),flush=True)
            rows.extend(new)
    counts = Counter(int(r["dsid"]) for r in rows)
    if counts!=PAIR_COUNTS:
        raise ValueError(f"Old positive pairs do not reproduce the published folds: {counts}")
    out = RUN/"old_positive_higgs_jet_decays.csv"
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w",newline="") as stream:
        w = csv.DictWriter(stream,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out)


if __name__=="__main__":
    main()
