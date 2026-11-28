"""Review the effective Higgs supervision and the named WH case in fold 1 training."""
import json
from collections import Counter, defaultdict
import numpy as np
import uproot
from paths import BASELINE, PLAN, RUN
from publish import I,J,matrix


def previous_hh_evidence(plan):
    parts=defaultdict(list)
    jobs=(j for j in plan["jobs"] if j["kind"]=="weights")
    for job in sorted(jobs,key=lambda j:(j["sample"],j["id"])):
        sample=plan["samples"][job["sample"]]
        if not sample["is_signal"] or sample["is_data"]: continue
        with np.load(BASELINE/"raw_labels"/f"{job['id']:06}.npz") as f:
            if "base_id" not in f: continue
            n=f["base_id"]
            keep=n%3==0
            parts[int(sample["dsid"])].append((n[keep],f["base_bits"][keep,:3]))
    return {d:(np.concatenate([r[0] for r in rows]),np.concatenate([r[1] for r in rows]))
            for d,rows in parts.items()}


def main():
    audit=json.loads((RUN/"dataset/audit.json").read_bytes())["splits"]["base_0"]
    summary=json.loads((RUN/"dataset/summary.json").read_bytes())
    plan=json.loads(PLAN.read_bytes())
    samples=plan["samples"].values()
    hh_bits=previous_hh_evidence(plan)
    offsets=Counter()
    descriptions={int(s["dsid"]):s["description"] for s in samples if not s["is_data"]}
    sow=np.array(summary["sum_abs_weights_by_remainder"][0])
    factors=sow[1]/sow
    wh=Counter()
    lost=defaultdict(Counter)
    lost_by_origin = Counter()
    example = None
    effective=defaultdict(lambda:np.zeros(2))
    both=["dsid","eventNumber","ancestryHiggs","relationHiggs"]
    fields=[*both,"valid","isMET","isPrompt","partOrigin","weight","class_id","label_signal",
            "higgsPairKnown","higgsPairSame"]
    def tally(key,values,weights):
        known=(values>=0).sum(1)
        use=known>0
        effective[key]+=(np.dot(weights[use],(values[use]==1).sum(1)/known[use]),weights[use].sum())
    with uproot.open(RUN/"dataset/3fold_0.root") as current,uproot.open(BASELINE/"dataset/3fold_0.root") as previous:
        new,old=current["tree"],previous["tree"]
        for start in range(0,new.num_entries,48000):
            stop=min(start+48000,new.num_entries)
            a=new.arrays(fields,entry_start=start,entry_stop=stop,library="np")
            b=old.arrays([*both,"relationW","relationZ"],entry_start=start,entry_stop=stop,library="np")
            for key in ("dsid","eventNumber"): np.testing.assert_array_equal(a[key],b[key])
            valid=a["valid"].astype(bool)&~a["isMET"].astype(bool)
            select=np.isin(a["dsid"],(604305,604306))
            if select.any():
                new_lep=a["ancestryHiggs"][select,:3]
                old_lep=b["ancestryHiggs"][select,:3]
                prompt_w=a["isPrompt"][select,:3].astype(bool)&(a["partOrigin"][select,:3]==4)
                wh["events"]+=int(select.sum())
                for tag,code in (("unknown",-1),("negative",0),("positive",1)):
                    wh["new_"+tag]+=int((new_lep==code).sum())
                    wh["old_"+tag]+=int((old_lep==code).sum())
                wh["new_negative_prompt_W_origin"]+=int(((new_lep==0)&prompt_w).sum())
            hh=a["label_signal"].astype(bool)
            for dsid in np.unique(a["dsid"][hh]):
                keep=hh & (a["dsid"]==dsid)
                positives=b["ancestryHiggs"][keep,:3]==1
                rows=slice(offsets[dsid],offsets[dsid]+int(keep.sum()))
                n,bits=hh_bits[dsid]
                np.testing.assert_array_equal(n[rows],a["eventNumber"][keep])
                offsets[dsid]=rows.stop
                direct=((bits[rows]&12)!=0)|(b["relationW"][keep,:3]>0)|(b["relationZ"][keep,:3]>0)
                became_unknown=positives&(a["ancestryHiggs"][keep,:3]<0)
                raw_higgs = (bits[rows] & 1) != 0
                old_id = b["relationHiggs"][keep,:3]
                lost[int(dsid)]["affected_events"] += int(became_unknown.any(axis=1).sum())
                lost[int(dsid)]["now_unknown_with_raw_Higgs_mask"] += int((became_unknown & raw_higgs).sum())
                lost[int(dsid)]["now_unknown_with_old_Higgs_ID"] += int((became_unknown & (old_id > 0)).sum())
                if np.any(became_unknown & ~(raw_higgs | (old_id > 0) | direct)):
                    raise ValueError("Unexplained previous positive Higgs label")
                origins = a["partOrigin"][keep,:3]
                for origin, count in zip(*np.unique(origins[became_unknown], return_counts=True)):
                    lost_by_origin[int(origin)] += int(count)
                if dsid == 604422 and example is None and became_unknown.any():
                    row, slot = np.argwhere(became_unknown)[0]
                    example = dict(dsid=int(dsid), eventNumber=int(n[rows][row]), lepton_slot=int(slot + 1),
                                   raw_Higgs_mask_positive=bool(raw_higgs[row,slot]), old_Higgs_ID=int(old_id[row,slot]),
                                   old_direct_WZ_match=bool(direct[row,slot]), partOrigin=int(origins[row,slot]),
                                   old_ancestry=1, new_ancestry=-1)
                lost[int(dsid)]["old_positive"]+=int(positives.sum())
                lost[int(dsid)]["now_unknown"]+=int(became_unknown.sum())
                lost[int(dsid)]["old_positive_with_direct_WZ_evidence"]+=int((positives&direct).sum())
                lost[int(dsid)]["now_unknown_with_direct_WZ_evidence"]+=int((became_unknown&direct).sum())
            w=np.abs(a["weight"].astype("f8"))*factors[a["class_id"]]
            old_pair=matrix(b["relationHiggs"])
            new_pair=np.where(((a["higgsPairKnown"][:,I]>>J)&1)!=0,
                              ((a["higgsPairSame"][:,I]>>J)&1).astype("i1"),-1)
            for name,y,pairs in (("previous",b["ancestryHiggs"],old_pair),("new",a["ancestryHiggs"],new_pair)):
                tally(name+"_ancestry",np.where(valid,y,-1),w)
                tally(name+"_pairs",np.where(valid[:,I]&valid[:,J],pairs,-1),w)
    if any(offsets[d]!=len(pair[0]) for d,pair in hh_bits.items()):
        raise RuntimeError("Incomplete HH evidence coverage")
    hh_total = Counter()
    for counts in lost.values():
        hh_total.update(counts)
    result=dict(inclusive_WH_leptons=dict(wh),fully_matched_HH=audit["fully_matched_HH_old_labels_by_new"],
                HH_lepton_total=dict(hh_total),
                HH_unknown_by_partOrigin=dict(sorted(lost_by_origin.items())), HH_lepton_example=example,
                effective_positive_fraction={key:v[0]/v[1] for key,v in effective.items()},
                positive_HH_leptons_now_unknown_by_dsid={str(d):dict(description=descriptions[d],**v) for d,v in lost.items()})
    (RUN/"label_review.json").write_text(json.dumps(result,indent=2)+"\n")
    print("WH:",dict(wh),"effective positive weight:",result["effective_positive_fraction"])
    for d,v in sorted(lost.items(), key=lambda item:-item[1]["now_unknown"]):
        print(d,descriptions[d],dict(v))


if __name__=="__main__":
    main()
