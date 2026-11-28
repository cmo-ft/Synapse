"""Check lepton pair assignments and a truth-mass fallback in selected continuum diboson MC."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import awkward as ak
import numpy as np
import uproot


RECIPE = Path(__file__).resolve().parent
HERE = RECIPE.parents[1] / "run_dir" / RECIPE.name
CAF = HERE.parents[3] / "caf/hhml3l0tau"
PAIRS = ((0, 1), (0, 2), (1, 2))
SAMPLES = {"701045_c23a": "lllv", "701005_c23a": "lllvjj", "701040_c23a": "llll",
           "701105_c23a": "WqqZll"}
Z_MASS, NEAR_Z, FAR_Z = 91.2, 5.0, 15.0  # GeV


def event_key(number, pts, pdgs):
    return int(number), tuple(sorted(zip(map(float, pts), map(int, pdgs))))


def strict_z_mass_choice(ids, p4):
    """Test both OS assignments, withholding flavour for the mixed-flavour cross-check."""
    positive = [i for i, pdg in enumerate(ids) if pdg > 0]
    negative = [i for i, pdg in enumerate(ids) if pdg < 0]
    if len(ids) == 3:
        options = [((i, j),) for i in positive for j in negative]
    elif len(ids) == 4 and len(positive) == len(negative) == 2:
        options = [list(zip(positive, negative)), list(zip(positive, reversed(negative)))]
    else:
        return None
    if len(options) != 2:
        return None
    scored = []
    for pairs in options:
        offsets = []
        for i, j in pairs:
            e, px, py, pz = (a + b for a, b in zip(p4[i], p4[j]))
            m2 = e*e - px*px - py*py - pz*pz
            if not math.isfinite(m2):
                return None
            offsets.append(abs(math.sqrt(max(0., m2)) * .001 - Z_MASS))
        scored.append((max(offsets), pairs))
    scored.sort(key=lambda score: score[0])
    (best, chosen), (rival, _) = scored
    if best < NEAR_Z and rival > FAR_Z:
        return all(ids[i] == -ids[j] for i, j in chosen)
    return None


def raw_matches(path, mode, *, entry_stop=8192, event_numbers=None):
    fields = ["eventNumber", "hhml_pass_3l_NOSYS"]
    for kind, label in (("el", "Electron"), ("mu", "Muon")):
        fields.extend(f"{kind}_{x}" for x in ("pt_NOSYS", "charge", "eta", "phi", f"isAnalysis{label}_NOSYS"))
        fields.extend(f"truth{label.lower()}_{x}" for x in ("pdgId", "px", "py", "pz", "e"))
    summary = {}
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
        tree = f["AnalysisMiniTree"]
        branches = [k for k in tree.keys() if k.startswith(("truthelectron_", "truthmuon_"))]
        for data in tree.iterate(fields, entry_stop=entry_stop, step_size=50000, library="ak"):
            data = data[(data.eventNumber % 3 == 2) & (data.hhml_pass_3l_NOSYS != 0)]
            if event_numbers is not None:
                data = data[np.isin(ak.to_numpy(data.eventNumber), event_numbers)]
            for key, details in matched_rows(data, mode):
                summary[key] = details if key not in summary else None
    return summary, branches


def matched_rows(data, mode):
    for row in ak.to_list(data):
        reco, truth, momenta = [], [], []
        for kind, label, flavor in (("el", "Electron", 11), ("mu", "Muon", 13)):
            reco.extend((-flavor * int(row[kind+"_charge"][i]), row[kind+"_pt_NOSYS"][i],
                         row[kind+"_eta"][i], row[kind+"_phi"][i])
                        for i, selected in enumerate(row[f"{kind}_isAnalysis{label}_NOSYS"]) if selected)
            prefix = "truth" + label.lower() + "_"
            for pdg, px, py, pz, energy in zip(*(row[prefix + k] for k in ("pdgId", "px", "py", "pz", "e"))):
                pt = math.hypot(px, py)
                eta = math.asinh(pz/pt) if pt > 0 else math.nan
                truth.append((int(pdg), eta, math.atan2(py, px)))
                momenta.append((energy, px, py, pz))
        if len(reco) != 3:
            continue
        links = [[i for i, (tpdg, teta, tphi) in enumerate(truth) if pdg == tpdg and
                  math.hypot(eta-teta, math.remainder(phi-tphi, 2*math.pi)) < 0.1] for pdg, pt, eta, phi in reco]
        matched = all(len(link) == 1 for link in links) and len({link[0] for link in links}) == 3
        truth_ids = sorted(t[0] for t in truth)
        hypotheses = sum(a == -b for a, b in combinations(truth_ids, 2))
        label = None
        if matched and mode.startswith("lllv") and len(truth) == 3 and hypotheses in (1, 2):
            label = "unique" if hypotheses == 1 else "ambiguous"
        if matched and mode == "llll" and len(truth) == 4:
            if truth_ids == [-13, -11, 11, 13]:
                label = "unique"
            elif truth_ids in ([-13, -13, 13, 13], [-11, -11, 11, 11]):
                label = "ambiguous"
        key = event_key(row["eventNumber"], [r[1] for r in reco], [r[0] for r in reco])
        mass_choice = strict_z_mass_choice([t[0] for t in truth], momenta) if label else None
        yield key, (matched, label, mass_choice)


def selected_counts(path, region, raw):
    fields = ["eventNumber", "n_jet"] + [f"l{i}_{k}" for i in (1, 2, 3)
                                        for k in ("pt", "pdg", "isPrompt", "truthOrigin")]
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as f:
        a = f[region].arrays(fields, entry_stop=30000, library="np")
    keep = (a["eventNumber"] % 3 == 2) & (a["n_jet"] >= 1)
    jets = a["n_jet"][keep]

    def leptons(field):
        return np.stack([a[f"l{i}_{field}"][keep] for i in (1, 2, 3)], axis=1)
    pts, pdgs, prompt, origin = (leptons(k) for k in ("pt", "pdg", "isPrompt", "truthOrigin"))
    prompt = prompt.astype(bool)
    all_prompt = prompt.all(1)
    sfos = np.stack([pdgs[:, i] == -pdgs[:, j] for i, j in PAIRS], axis=1)
    two_prompt = np.stack([prompt[:, i] & prompt[:, j] for i, j in PAIRS], axis=1)
    counts = dict(selected=int(keep.sum()), all_prompt=int(all_prompt.sum()),
                  all_prompt_origins_43=int((all_prompt & (origin == 43).all(1)).sum()),
                  all_prompt_one_sfos=int((all_prompt & (sfos.sum(1) == 1)).sum()),
                  all_prompt_two_sfos=int((all_prompt & (sfos.sum(1) == 2)).sum()),
                  two_prompt_sfos=int(((prompt.sum(1) == 2) & (two_prompt & sfos).any(1)).sum()))
    joined, seen = Counter(), set()
    mass_counts = {"mixed_flavour_control": Counter(), "same_flavour_ambiguous": Counter()}
    for i, number in enumerate(a["eventNumber"][keep]):
        if not all_prompt[i]:
            continue
        key = event_key(number, pts[i], pdgs[i])
        details = raw.get(key)
        if details is None or key in seen:
            continue
        seen.add(key)
        matched, label, mass_choice = details
        joined["caf_prompt_in_raw_prefix"] += 1
        joined["three_unique_truth_matches"] += int(matched)
        if label:
            joined["exclusive_truth_content_"+label] += 1
            if jets[i] >= 2:
                joined["exclusive_truth_content_"+label+"_at_least_two_jets"] += 1
            counts_for_label = mass_counts["mixed_flavour_control" if label == "unique" else "same_flavour_ambiguous"]
            counts_for_label["eligible"] += 1
            if mass_choice is not None:
                counts_for_label["assigned"] += 1
                if label == "unique":
                    counts_for_label["agree_with_flavour"] += int(mass_choice)
    counts["raw_truth_crosscheck"] = dict(joined)
    counts["truth_mass_check"] = {k: dict(v) for k, v in mass_counts.items()}
    return counts


def main():
    plan = json.loads((CAF/"work/nominal-run3-20261122/plan/plan.json").read_text())
    results = {}
    for sample, mode in SAMPLES.items():
        job = next(j for j in plan["jobs"] if j["sample"] == sample and j["kind"] == "weights")
        raw, fields = raw_matches(job["files"][0], mode)
        regions = {}
        for reg in ("sr", "wzcr"):
            path = CAF/"output/nominal-run3-20261122/ntuples/run3/parts/NOSYS"/sample/f"{job['id']:06}_{reg}.root"
            regions[reg] = selected_counts(path, reg, raw)
        results[sample] = dict(mode=mode, description=plan["samples"][sample]["description"], job=job["id"],
                               raw_file=job["files"][0], truth_lepton_fields=fields, regions=regions)
    report = dict(scope="One c23a CAF weights job per DSID, first up to 30,000 CAF rows per region, training "
                  "remainder 2, >=1 central jet. WZCR before one-in-eight sampling. Raw crosscheck uses the first "
                  "up to 8,192 entries of the first associated EJ file, joined by event number and three signed PDGs "
                  "and reco pT. Counts are unweighted.", match="All three reconstructed leptons must match distinct "
                  "truth e/μ, each with exactly one candidate at the same signed PDG and ΔR < 0.1. Strict topology "
                  "counts additionally require exactly 3 (WZ) or 4 (ZZ) recorded truth e/μ. The raw_truth_crosscheck "
                  "pair assignments use charge and flavour only.",
                  truth_mass_rule=dict(z_mass_GeV=Z_MASS, window_GeV=NEAR_Z, alternative_outside_GeV=FAR_Z,
                  definition="Use truth e,px,py,pz. WZ ranks the two OS pairs. ZZ ranks the two complete OS "
                  "pairings of four truth leptons by their larger pole-mass deviation. Assign only if the best "
                  "deviation is inside the window and the other is outside the veto. In mixed-flavour events the "
                  "choice hides flavour, then tests agreement with the unique SFOS solution. Same-flavour events "
                  "have no independent identity label in this study."),
                  samples=results)
    HERE.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=HERE, delete=False) as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    Path(f.name).replace(HERE/"continuum_pair_audit.json")
    for name, row in results.items():
        print(name, row["regions"], flush=True)


if __name__ == "__main__":
    main()
