"""Trace both fold-1 HH fallback groups through the independent EasyJet truth-tau collection."""
import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import chain
from typing import NamedTuple

import awkward as ak
import numpy as np
import uproot

from paths import BASELINE, PHYSICS, PLAN, RELATIONS, RUN


class Particle(NamedTuple):
    pdg: int
    pt: float
    eta: float
    phi: float


class Cuts(NamedTuple):
    visible_dr: float
    visible_relative_pt: float
    reco_dr: float
    reco_relative_pt: float


CUTS = {
    "tight": Cuts(0.001, 0.01, 0.05, 0.25),
    "nominal": Cuts(0.01, 0.10, 0.05, 0.35),
    "loose": Cuts(0.05, 0.30, 0.10, 0.50),
}
ORIGIN_CODES = {"H_tau": 14, "W_tau": 12, "Z_tau": 13}


def from_xyz(pdg, x, y, z):
    pt = math.hypot(x, y)
    return Particle(pdg, pt, math.asinh(z / pt) if pt else math.inf, math.atan2(y, x))


def near(a, b, max_dr, max_rel_pt):
    return (b.pt > 0 and abs(a.pt / b.pt - 1) < max_rel_pt
            and math.hypot(a.eta - b.eta, math.remainder(a.phi - b.phi, 2 * math.pi)) < max_dr)


def higgs_daughters(event):
    tau_legs, direct_leptons = [], []
    for parent in (1, 2):
        if event[f"truth_H{parent}_pdgId"] != 25:
            continue
        def vectors(generation):
            return [event[f"truth_{generation}_fromH{parent}_{key}"] for key in ("pdgId", "pt", "eta", "phi")]
        grandchildren = vectors("grandchildren")
        for index, child in enumerate(map(Particle._make, zip(*vectors("children"), strict=True))):
            particle = abs(child.pdg)
            if particle == 15:
                tau_legs.append(("H_tau", child))
            elif particle in (11, 13):
                direct_leptons.append(child)
            if particle not in (23, 24) or any(index >= len(field) for field in grandchildren):
                continue
            for daughter in map(Particle._make, zip(*(v[index] for v in grandchildren), strict=True)):
                if abs(daughter.pdg) == 15 and (particle == 23 or child.pdg * daughter.pdg < 0):
                    tau_legs.append(("W_tau" if particle == 24 else "Z_tau", daughter))
                elif abs(daughter.pdg) in (11, 13):
                    direct_leptons.append(daughter)
    return tau_legs, direct_leptons


def truth_objects(event):
    leptons = []
    for prefix in ("truthelectron_", "truthmuon_"):
        leptons.extend(from_xyz(*values) for values in zip(
            *(event[prefix + key] for key in ("pdgId", "px", "py", "pz")), strict=True))
    tau_fields = ("px", "py", "pz", "IsHadronicTau", "pt_vis", "eta_vis", "phi_vis", "classifierParticleOrigin")
    taus = []
    for x, y, z, hadronic, pt, eta, phi, origin in zip(*(event["truthtau_" + k] for k in tau_fields), strict=True):
        taus.append((from_xyz(0, x, y, z), bool(hadronic), Particle(0, pt, eta, phi), origin))
    return taus, leptons


def match_tau_legs(legs, taus):
    # The full truth-tau momentum should identify the same tau in the two collections.
    links = [(i, j) for i, (_, leg) in enumerate(legs) for j, tau in enumerate(taus)
             if near(leg, tau[0], 0.001, 0.01)]
    n_leg, n_tau = Counter(i for i, _ in links), Counter(j for _, j in links)
    return [(legs[i][0], legs[i][1].pdg, taus[j][2], taus[j][3]) for i, j in links
            if n_leg[i] == n_tau[j] == 1 and not taus[j][1]]


def match_visible(tau_links, truth, reco, cuts):
    # Each link must be unique at both ends, including lepton candidates outside the group being studied.
    tau_truth = [(i, j) for i, (_, tau_pdg, visible, _) in enumerate(tau_links) for j, lep in enumerate(truth)
                 if tau_pdg * lep.pdg > 0 and near(lep, visible, cuts.visible_dr, cuts.visible_relative_pt)]
    n_tau, n_truth = Counter(i for i, _ in tau_truth), Counter(j for _, j in tau_truth)
    unique_tau = {j: tau_links[i] for i, j in tau_truth if n_tau[i] == n_truth[j] == 1}
    truth_reco = [(j, k) for j, lep in enumerate(truth) for k, r in enumerate(reco)
                  if lep.pdg == r.pdg and near(r, lep, cuts.reco_dr, cuts.reco_relative_pt)]
    n_truth, n_reco = Counter(j for j, _ in truth_reco), Counter(k for _, k in truth_reco)
    return {k: unique_tau[j] for j, k in truth_reco if j in unique_tau and n_truth[j] == n_reco[k] == 1}


def input_branches():
    names = ["eventNumber"]
    for h in (1, 2):
        names.append(f"truth_H{h}_pdgId")
        for gen in ("children", "grandchildren"):
            names.extend(f"truth_{gen}_fromH{h}_{k}" for k in ("pdgId", "pt", "eta", "phi"))
    for prefix in ("truthelectron_", "truthmuon_"):
        names.extend(prefix + k for k in ("pdgId", "px", "py", "pz"))
    names.extend("truthtau_" + k for k in ("px", "py", "pz", "IsHadronicTau", "pt_vis", "eta_vis", "phi_vis",
                                            "classifierParticleOrigin"))
    for prefix, flavour in (("el_", "Electron"), ("mu_", "Muon")):
        names.extend(prefix + k for k in ("pt_NOSYS", "charge", f"isAnalysis{flavour}_NOSYS"))
    return names


def selected_rows(job):
    name = f"{job['id']:06}"
    with (np.load(RUN / "shards" / (name + ".npz")) as new,
          np.load(BASELINE / "raw_labels" / (name + ".npz")) as old,
          np.load(RELATIONS / "parent_shards" / (name + ".npz")) as mask):
        if "base_id" not in new:
            return None
        event = new["base_id"]
        for archive in (old, mask):
            np.testing.assert_array_equal(event, archive["base_id"])
            np.testing.assert_array_equal(new["base_pt"], archive["base_pt"])
        affected = (event % 3 == 0)[:, None] & (new["base_ancestry"][:, :3] < 0) & ((old["base_bits"][:, :3] & 1) != 0)
        rows = affected.any(axis=1)
        if not rows.any():
            return None
        mask_id = mask["base_ids"][rows, 0, :3]
        with uproot.open(PHYSICS / "shards" / (name + ".root"), handler=uproot.source.file.MemmapSource) as source:
            a = source["base"].arrays(["eventNumber", "pt", "pdg", "eta", "phi", "partOrigin", "isQMisID"], library="np")
        np.testing.assert_array_equal(event, a.pop("eventNumber"))
        np.testing.assert_array_equal(new["base_pt"], a["pt"][:, :3])
        objects = {key: values[rows, :3] for key, values in a.items()}
        return event[rows], affected[rows], mask_id, objects


def signature(event):
    tracks = []
    for obj, flavour, pdg in (("el", "Electron", 11), ("mu", "Muon", 13)):
        tracks.extend((pt, -pdg * round(charge)) for pt, charge, valid in zip(
            event[obj + "_pt_NOSYS"], event[obj + "_charge"], event[obj + "_isAnalysis" + flavour + "_NOSYS"], strict=True)
                      if valid)
    return sorted(tracks)


def check_event(event, reco, affected, mask_id, origins, charge_misid, counts):
    legs, direct = higgs_daughters(event)
    taus, truth = truth_objects(event)
    linked = match_tau_legs(legs, taus)
    matches = {name: match_visible(linked, truth, reco, cuts) for name, cuts in CUTS.items()}
    for i in np.flatnonzero(affected):
        origin = int(origins[i])
        group = "would_recover" if origin == 1 else "would_remain_unknown"
        c = counts[group]
        c["leptons"] += 1
        c["origin/" + str(origin)] += 1
        c["mask_unique" if mask_id[i] > 0 else "mask_multiple"] += 1
        if any(path == "W_tau" and tau.pdg * reco[i].pdg > 0 for path, tau in legs):
            c["compatible_W_tau_in_event"] += 1
        for profile, assignment in matches.items():
            source = assignment[i][0] if i in assignment else "unresolved"
            c[profile + "/" + source] += 1
        nominal = matches["nominal"].get(i)
        source = nominal[0] if nominal else "unresolved"
        c[f"origin_by_chain/{origin}/{source}"] += 1
        if nominal and nominal[3] == ORIGIN_CODES[source]:
            c["tau_origin_agrees/" + source] += 1
        if not nominal:
            if any(p.pdg == reco[i].pdg and near(reco[i], p, .1, .5) for p in direct):
                c["unresolved_close_to_explicit_non_tau_lepton"] += 1
            if charge_misid[i]:
                c["unresolved_charge_misid"] += 1


def run_job(job, counts):
    selected = selected_rows(job)
    if selected is None:
        return 0
    numbers, affected, mask_id, obj = selected
    by_number = defaultdict(list)
    for row, event in enumerate(numbers):
        by_number[int(event)].append(row)
    found = np.zeros(len(numbers), dtype=np.int8)
    for file in job["files"]:
        with uproot.open(file, handler=uproot.source.file.MemmapSource, array_cache=None) as source:
            tree = source["AnalysisMiniTree"]
            hits = np.isin(tree["eventNumber"].array(library="np"), numbers)
            for start in range(0, len(hits), 12000):
                keep = hits[start:start + 12000]
                if not keep.any():
                    continue
                batch = tree.arrays(input_branches(), entry_start=start, entry_stop=start + len(keep), library="ak")
                for event in ak.to_list(batch[keep]):
                    raw_key = signature(event)
                    rows = [row for row in by_number[event["eventNumber"]]
                            if sorted(zip(obj["pt"][row], obj["pdg"][row])) == raw_key]
                    if not rows:
                        continue
                    if len(rows) != 1 or found[rows[0]]:
                        raise ValueError(f"Ambiguous event {job['sample']} {event['eventNumber']}")
                    row, = rows
                    found[row] = 1
                    reco = [Particle(int(pdg), float(pt), float(eta), float(phi)) for pdg, pt, eta, phi
                            in zip(*(obj[k][row] for k in ("pdg", "pt", "eta", "phi")))]
                    check_event(event, reco, affected[row], mask_id[row], obj["partOrigin"][row],
                                obj["isQMisID"][row], counts)
    if not found.all():
        raise ValueError(f"Missing {int((found == 0).sum())} indexed events in {job['sample']}")
    print(job["sample"], len(numbers), "affected events", flush=True)
    return len(numbers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", help="Check one DSID/campaign without publishing the full totals")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_bytes())
    counts = defaultdict(Counter)
    events = 0
    for job in plan["jobs"]:
        sample = plan["samples"][job["sample"]]
        if job["kind"] == "weights" and not sample["is_data"] and sample["is_signal"]:
            if not args.sample or args.sample == job["sample"]:
                events += run_job(job, counts)
    if args.sample:
        print(json.dumps(counts, indent=2))
        return
    assert events == 71604
    assert counts["would_recover"]["leptons"] == 94217
    assert counts["would_remain_unknown"]["leptons"] == 14122
    for c in counts.values():
        for name in CUTS:
            assert sum(c[name + "/" + k] for k in chain(ORIGIN_CODES, ("unresolved",))) == c["leptons"]
    result = dict(events=events, counts=counts, cuts={name: c._asdict() for name, c in CUTS.items()},
                  selection="Unweighted fold 1 HH training leptons in SR + ExtConvCR, tree unknown and EasyJet mask positive.",
                  tau_identification="One-to-one H-record tau to full truth-tau match, deltaR < 0.001 and relative pT < 1%. "
                    "Require a leptonic truth tau, match its visible momentum to a same-sign truth electron or muon, "
                    "then match to a reconstructed lepton of the same flavour and charge. No tau daughter barcode is stored.")
    path = RUN / "tau_fallback_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
