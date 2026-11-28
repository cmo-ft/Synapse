"""Write the side-by-side summary once both fold evaluations and targeted probes exist."""
import csv
import json
from math import isclose

from paths import RUN, ANCESTRY_RUN


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def native(comparison, name):
    return comparison[name+"_score_"+name+"_veto"]


def arrow(a, b, digits=5):
    return f"{a:.{digits}f} → {b:.{digits}f}"


def observation(x, y, digits=3):
    return f"{float(x):+.{digits}f} ± {float(y):.{digits}f}"


def pairrow(values, **match):
    result = [r for r in values if all(r[k] == v for k,v in match.items())]
    if len(result) != 1:
        raise ValueError(f"Expected one row for {match}, got {len(result)}")
    return result[0]


def budget_section(folder):
    scores = folder/"same_epoch_budget.json"
    benchmark = folder/"same_epoch_budget_trafo60.json"
    if not scores.is_file() or not benchmark.is_file():
        return []
    budget = read(scores)
    trafo = [read(p)["folds"] for p in (benchmark, folder.parent/"hhml_eval/summary.json")]
    previous, current = budget["previous_epoch"], budget["current_epoch"]
    lines = ["## Fold 0 with the same checkpoint selection window", "",
             f"Both models could select any of their first {budget['max_epochs_considered']} epochs. "
             f"The best validation three-bin Z selected previous epoch {previous} and current epoch {current}. "
             "The previous checkpoint was reapplied on CPU. The current checkpoint uses its published scores.", "",
             "| Sample | Three-bin Z, previous → current | Inclusive R20, previous → current | Trafo60 Z, previous → current |",
             "|---|---:|---:|---:|"]
    for split, rem in (("validation", 1), ("test", 0)):
        result = budget["splits"][split]["same_split"]
        old, new = result["previous_budget"], result["current"]
        z = [v[f"fold_{rem}"]["models"]["Synapse"]["trafo60_z"] for v in trafo]
        lines.append(f"| {split.capitalize()} | {arrow(old['quadrature'], new['quadrature'])} | "
                     f"{arrow(old['R20'], new['R20'], 1)} | {arrow(*z)} |")
    transfer = budget["splits"]["test"]["validation_cuts"]
    delta = transfer["current"]["quadrature"] - transfer["previous_budget"]["quadrature"]
    error = transfer["paired_delta_z"]["sd"]
    lines.extend(("", "Applying each model’s validation thresholds directly to test gives "
                  f"ΔZ = {observation(delta, error, 5)} (current minus previous, paired MC uncertainty).", ""))
    return lines


def render():
    out = [RUN/f"output/fold_{i}/evaluation" for i in (0,1)]
    sub = [p/"ancestry_comparison" for p in out]
    if not all((p/"all_wz_recoil_probe.json").is_file() for p in sub):
        return
    metrics = [read(p/"event_comparison.json") for p in sub]
    text = ["# Previous and current Synapse checkpoints on both test folds", "",
            "Arrows give the 26 November ancestry model followed by the 27 November Higgs lepton fallback model. Each fold uses its own validation-selected checkpoint. The two test partitions contain different events. Each partition also serves another model in a different training or validation role.", ""]
    stop = RUN/"output/fold_0/early_stop.json"
    if stop.is_file():
        record = read(stop)
        text.extend((f"The current fold 0 fit ended at the requested boundary after {record['completed_epochs']} of 50 epochs and used epoch {record['selected_epoch']} for application. The previous fold 0 fit and both fold 1 fits ran all 50 epochs. The fold 0 comparison therefore includes the effect of the shorter training schedule.", ""))
    text.extend(("| Fold and sample | Three-bin Z at matched HH efficiency | Inclusive R20 | Common Trafo60 Z | EveNet Trafo60 Z |",
                 "|---|---:|---:|---:|---:|"))
    for i in (0,1):
        benchmarks = [read(run/f"output/fold_{i}/evaluation/hhml_eval/summary.json")["folds"] for run in (ANCESTRY_RUN,RUN)]
        for split, rem in (("validation", (i+1)%3), ("test", i)):
            methods = [b[f"fold_{rem}"]["models"] for b in benchmarks]
            eve_old, eve_new = [item["EveNet"] for item in methods]
            if (eve_old["trafo60_bins"] != eve_new["trafo60_bins"] or
                    not isclose(eve_old["trafo60_z"], eve_new["trafo60_z"], rel_tol=0, abs_tol=1e-12)):
                raise ValueError("EveNet reference changed")
            m = [native(metrics[i][split]["split"],name) for name in ("previous","current")]
            text.append(f"| {i} {split} | {arrow(*(x['quadrature'] for x in m))} | "
                        f"{arrow(*(x['positive_background_rejection'][0] for x in m),1)} | "
                        f"{arrow(*(x['Synapse']['trafo60_z'] for x in methods))} | {methods[0]['EveNet']['trafo60_z']:.5f} |")
    text += ["", "Three-bin Z combines the disjoint 0–20%, 20–30% and 30–50% HH-efficiency bands. Trafo60 fits and evaluates edges separately per split using the common evaluator.", ""]
    text += budget_section(sub[0])
    text += ["| Test measurement | Fold 0 | Fold 1 |", "|---|---:|---:|"]
    cell = []
    for i in (0,1):
        x = metrics[i]["test"]["validation"]
        dz = native(x,"current")["quadrature"] - native(x,"previous")["quadrature"]
        cell.append(observation(dz, x["paired_delta_z_bootstrap"]["sd"],5))
    text.append(f"| Δ three-bin Z at frozen validation cuts | {' | '.join(cell)} |")
    migration = [rows(p/"process_migrations.csv") for p in sub]
    for name in ("all backgrounds", "WZ", "Other VV", "VVV", "Vjets", "Vy", "top", "singleH"):
        cells = [pairrow(x, split="test",cuts="split",band_type="cumulative",wp="20",process=name) for x in migration]
        text.append(f"| Δ expected {name} at 20% | "+" | ".join(observation(r["delta"],r["paired_mc_error"]) for r in cells)+" |")
    text += ["", "Δ is current minus previous. Each process error uses paired MC migrations at the observed thresholds. The Z error uses paired Poisson resampling on test with fixed validation cuts.", "",
             "## How the event head uses WZ recoil", "",
             "| Test WZ yield difference at matched 20% HH efficiency | Fold 0 | Fold 1 |","|---|---:|---:|"]
    recoil = [rows(p/"all_wz_recoil_paired_changes.csv") for p in sub]
    for key, label in (("current_minus_previous_native","Current minus previous with native recoil"),
                       ("current_minus_previous_no_recoil","Current minus previous with recoil off"),
                       ("previous_wz_removed_by_native_recoil","WZ removed by the previous model’s recoil pathway"),
                       ("current_wz_removed_by_native_recoil","WZ removed by the current model’s recoil pathway"),
                       ("current_minus_previous_wz_removed_by_recoil","Current minus previous recoil benefit"),
                       ("previous_substitute_other_recoil","Change when the previous model takes the current recoil scalar"),
                       ("current_substitute_other_recoil","Change when the current model takes the previous recoil scalar")):
        cells = [pairrow(x, comparison=key,target_hh_efficiency="0.2") for x in recoil]
        text.append(f"| {label} | "+" | ".join(observation(r["wz_yield_change"],r["paired_wz_mc_error"]) for r in cells)+" |")
    text += ["", "Each comparison re-evaluates all HH and WZ test events. The receiving model’s event head and projection are frozen. The positive yield removed by recoil is the yield with recoil off minus the native yield.", "",
             "## Correct best same-Higgs pair within a test HH event", "",
             "| Pair type | Fold 0, previous → current | Fold 1, previous → current |","|---|---:|---:|"]
    assignments = [rows(p/"common_truth_best_pair.csv") for p in sub]
    for kind, label in (("ll","Lepton–lepton"),("lj","Lepton–jet"),("jj","Jet–jet")):
        cells = [pairrow(x,reference="tree_HH",objects=kind) for x in assignments]
        text.append(f"| {label} | "+" | ".join(arrow(100*float(r["previous_accuracy"]),100*float(r["current_accuracy"]),2)+"%" for r in cells)+" |")
    text += ["", "Both classifiers must choose among the same tree-resolved candidates. Eligible events have at least one correct candidate and one wrong candidate of that type.", ""]
    text += ["## W and Z pairs", "",
             "Both datasets use the same W and Z parent IDs, eligible entries and coefficients. The predicted matrices are learned again through the shared transformer.", "",
             "| Test measurement | Fold 0, previous → current | Fold 1, previous → current |",
             "|---|---:|---:|"]
    auxiliary = [[read(run/f"output/fold_{fold}/evaluation/test/auxiliary.json") for run in (ANCESTRY_RUN, RUN)]
                 for fold in (0, 1)]
    for label, measure in (("Correct highest-scoring W jet pair", "w_best"), ("W pair ROC AUC", "pair_w"), ("Z pair ROC AUC", "pair_z")):
        result = []
        for models in auxiliary:
            if measure == "w_best":
                old, new = [a["w_best_central_jet_pair"] for a in models]
                if old["eligible_events"] != new["eligible_events"]:
                    raise ValueError("The W assignment samples differ")
                result.append(arrow(100*old["weighted_accuracy"], 100*new["weighted_accuracy"], 1)+"%")
            else:
                old, new = [a["tasks"][measure] for a in models]
                if (old["positive"], old["negative"]) != (new["positive"], new["negative"]):
                    raise ValueError(f"The {measure} truth counts differ")
                result.append(arrow(old["auc_4096_bins"], new["auc_4096_bins"], 5))
        text.append(f"| {label} | "+" | ".join(result)+" |")
    text += ["", "The W assignment uses events with a verified hadronic W, a correct central jet pair and at least one other eligible pair. Each model ranks the same candidates. All three measures use absolute nominal MC weights.", ""]
    root = str(RUN).replace("/eos/home-c/cmo/", "/eos/user/c/cmo/", 1)
    text += ["## Background rejection curves", "",
             "Current NN, EveNet, XGBoost and previous NN use the original background groups. The lower panels give the ratio to the current NN at matched HH efficiency. Pages 1–4 show test, 5–8 show validation and 9–10 give the 20% working-point tables.", "",
             f"`{root}/rejection_comparison/background_rejection_comparison.pdf`", ""]
    for i in (0,1):
        base = str(sub[i]).replace("/eos/home-c/cmo/", "/eos/user/c/cmo/", 1)
        text.extend((f"### Fold {i} files", "", "```"))
        text.extend(f"{base}/{name}" for name in ("event_comparison.json", "background_yield_change.pdf",
                    "common_truth_best_pair.csv", "wz_recoil_test.pdf"))
        text.extend(("```", ""))
    dest = RUN/"test_comparison_by_fold.md"
    temp = dest.with_suffix(".md.tmp")
    temp.write_text("\n".join(text))
    temp.replace(dest)


if __name__ == "__main__":
    render()
