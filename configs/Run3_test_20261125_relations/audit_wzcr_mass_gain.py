"""Count new WZ Z-pair mass assignments on the actual fold-0, one-in-eight control sample."""
from collections import Counter
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import uproot

from audit_continuum_pairs import CAF, FAR_Z, HERE, NEAR_Z, PAIRS, Z_MASS, event_key, raw_matches


PREVIOUS = HERE.parent / "Run3_test_20261124_physics"
DSIDS = (701005, 701045)


def count_job(job, dsid):
    with uproot.open(PREVIOUS/"shards"/f"{job['id']:06}.root", handler=uproot.source.file.MemmapSource) as f:
        a = f["control"].arrays(["dsid", "eventNumber", "pt", "pdg", "isPrompt", "zPair", "weight"], library="np")
    if not np.all(a["dsid"] == dsid):
        raise ValueError("Unexpected DSID in prepared control shard")
    use = a["eventNumber"] % 3 == 2
    z, weight = a["zPair"][use], a["weight"][use].astype(np.float64)
    ids, pts, pdgs = a["eventNumber"][use], a["pt"][use, :3], a["pdg"][use, :3]
    prompt = a["isPrompt"][use, :3].astype(bool).all(1)
    sfos = np.stack([pdgs[:,i] == -pdgs[:,j] for i,j in PAIRS], axis=1)
    old = (z >= 0) & (z < 3)
    candidates = (z < 0) & prompt & (sfos.sum(1) == 2)
    target_rows = {}
    for i in np.flatnonzero(candidates):
        target_rows.setdefault(event_key(ids[i], pts[i], pdgs[i]), []).append(i)
    numbers = np.unique(ids[candidates])
    matches = {}
    mode = "lllvjj" if dsid == 701005 else "lllv"
    for source in job["files"]:
        if len(numbers):
            found, _ = raw_matches(source, mode, entry_stop=None, event_numbers=numbers)
            for key, detail in found.items():
                if key in target_rows:
                    matches[key] = detail if key not in matches else None
    added = np.zeros(len(z), dtype=bool)
    c = Counter(retained=len(z), existing_pairs=int(old.sum()), candidate_rows=int(candidates.sum()))
    for key, rows in target_rows.items():
        if key not in matches:
            c["missing_source_rows"] += len(rows)
            continue
        detail = matches[key]
        if detail is None:
            c["ambiguous_source_key_rows"] += len(rows)
            continue
        c["located_rows"] += len(rows)
        matched, label, assignment = detail
        if matched:
            c["three_reco_truth_matches"] += len(rows)
        if label == "ambiguous":
            c["exact_three_truth_two_sfos"] += len(rows)
            if assignment is not None:
                if not assignment:
                    raise ValueError("Same-flavour event yielded a different-flavour pair")
                added[rows] = True
        elif label is not None:
            raise ValueError("Reco and truth flavour assignments disagree")
    if np.any(added & old) or np.any(added & ~candidates):
        raise ValueError("Fallback overlaps existing supervision")
    c["additional_pairs"] = int(added.sum())
    c["combined_pairs"] = int(old.sum() + added.sum())
    for name, mask in (("existing", old), ("additional", added)):
        weights = weight[mask]
        c[name+"_sum_abs_weight"] = float(np.abs(weights).sum())
        c[name+"_sum_weight_squared"] = float(np.square(weights).sum())
    return dict(sample=job["sample"], dsid=dsid, job=job["id"], source_files=len(job["files"]), counts=dict(c))


def write_result(path, report):
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    Path(f.name).replace(path)


def main():
    plan = json.loads((CAF/"work/nominal-run3-20261122/plan/plan.json").read_text())
    HERE.mkdir(parents=True, exist_ok=True)
    output = HERE/"wzcr_mass_gain.json"
    samples = {f"{dsid}_{campaign}" for dsid in DSIDS for campaign in ("c23a", "c23d", "c23e")}
    jobs = [j for j in plan["jobs"] if j["kind"] == "weights" and j["sample"] in samples]
    results = {}
    for job in sorted(jobs, key=lambda j: j["id"]):
        dsid = int(job["sample"].split("_")[0])
        row = count_job(job, dsid)
        results[str(job["id"])] = row
        print(job["sample"], job["id"], row["counts"], flush=True)
    by_dsid, total = {}, Counter()
    for dsid in DSIDS:
        counts = Counter()
        for row in results.values():
            if row["dsid"] == dsid:
                counts.update(row["counts"])
        by_dsid[str(dsid)] = dict(counts)
        total.update(counts)
    with uproot.open(PREVIOUS/"dataset/wzcr_2.root", handler=uproot.source.file.MemmapSource) as f:
        a = f["tree"].arrays(["eventNumber", "dsid", "zPair", "weight"], library="np")
    if not np.all(a["eventNumber"] % 3 == 2):
        raise ValueError("Unexpected fold membership")
    old = (a["zPair"] >= 0) & (a["zPair"] < 3)
    for dsid in DSIDS:
        mask = a["dsid"] == dsid
        counts = by_dsid[str(dsid)]
        assert counts["retained"] == mask.sum() and counts["existing_pairs"] == (mask & old).sum()
        np.testing.assert_allclose(counts["existing_sum_abs_weight"], np.abs(a["weight"][mask & old]).sum(dtype=np.float64))
    report = dict(scope="Actual retained MC training control rows for fold 0 (eventNumber % 3 == 2). "
                  "Only the continuum WZ DSIDs 701005 and 701045 in all three Run 3 campaigns. "
                  "Read existing zPair assignments from the frozen shards and cross-check against wzcr_2.root. "
                  "Add a label only for a previously unlabelled three-prompt, two-SFOS row with three unique "
                  "charge/flavour-compatible ΔR < 0.1 reco matches and exactly three recorded truth e/μ. "
                  "The strictly isolated pole-mass assignment is auxiliary, not generator-parent truth. "
                  "Prepared data and training are untouched.", mass_GeV=dict(pole=Z_MASS, inside=NEAR_Z, rival_outside=FAR_Z),
                  fold_control_rows=len(old), fold_control_existing_nonnull_pairs=int(old.sum()),
                  fold_control_existing_null=int((a["zPair"] == 3).sum()), by_dsid=by_dsid, total=dict(total), jobs=results)
    write_result(output, report)
    print("total", dict(total), flush=True)


if __name__ == "__main__":
    main()
