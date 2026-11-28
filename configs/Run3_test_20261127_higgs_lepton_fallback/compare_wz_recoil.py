"""Measure the WZ recoil branch on every test HH and WZ event at equal HH efficiencies."""
import argparse
from contextlib import ExitStack
import csv
import json

import awkward as ak
import numpy as np
import torch

from paths import RUN
from probe_higgs_feedback import INPUTS, MODELS, open_tree, raw_inputs, read_checkpoint
from process_catalog import process_codes
from synapse.core.tools import extract_fields_from_expr

POINTS = (0.20, 0.30, 0.50)
MODES = ("native", "no_recoil", "other_recoil")


def thresholds(p, is_hh, w, pre):
    hh = is_hh & pre & (w > 0)
    allowed = np.flatnonzero(hh & (p[:, 1] < .8))
    ordered = allowed[np.argsort(-p[allowed, 0], kind="stable")]
    index = np.searchsorted(w[ordered].cumsum(), w[hh].sum()*np.asarray(POINTS))
    if index[-1] >= len(ordered):
        raise ValueError("Conversion veto prevents requested HH efficiency")
    return p[ordered[index], 0]


def save_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot(values, deltas, folder, fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11.3, 5.7), sharex="col", height_ratios=(2.25, 1))
    modes = ("native", "no_recoil", "other_recoil")
    x = np.arange(3)
    for col, eff in enumerate(POINTS):
        upper, lower = axes[:, col]
        for model, title, offset, color in (("previous","Previous",-.09,"#4879a7"), ("current","Current",.09,"#ae5037")):
            r = {d["recoil"]: d for d in values if d["model"] == model and float(d["target_hh_efficiency"]) == eff}
            upper.errorbar(x+offset, [float(r[m]["signed_wz"]) for m in modes],
                           yerr=[float(r[m]["mc_wz_error"]) for m in modes], fmt="o", color=color, ms=5, capsize=3, label=title)
        diffs = {d["comparison"]: d for d in deltas if float(d["target_hh_efficiency"]) == eff}
        r = [diffs["current_minus_previous_"+m] for m in modes]
        lower.errorbar(x, [float(d["wz_yield_change"]) for d in r], yerr=[float(d["paired_wz_mc_error"]) for d in r],
                       color="#273f50", fmt="o", ms=4, capsize=3)
        lower.axhline(0, color=".45", lw=.85)
        upper.set_title(f"{eff:.0%} HH efficiency")
        upper.set_ylim(bottom=0)
        upper.grid(axis="y", alpha=.17)
        lower.grid(axis="y", alpha=.17)
        lower.set_xticks(x, ("Own\nrecoil", "Recoil\noff", "Other model’s\nrecoil"))
        lower.set_xlim(-.52, 2.52)
    axes[0,0].set_ylabel("Expected WZ events")
    axes[1,0].set_ylabel("Current − previous")
    axes[0,0].legend(frameon=False)
    fig.suptitle(f"WZ rejection with the recoil branch changed during application  ·  Fold {fold} test")
    fig.text(.085, .025, "Each case uses all test HH and WZ at matched HH efficiency. The lower panels show paired MC statistical errors at the observed cuts.", fontsize=9)
    fig.tight_layout(rect=[0,.055,1,.965])
    fig.savefig(folder/"wz_recoil_test.pdf")
    plt.close(fig)


@torch.inference_mode()
def score_events(fold):
    nets, config, fcin = {}, {}, {}
    for key, path in MODELS.items():
        config[key], nets[key], _, _ = read_checkpoint(path / f"output/fold_{fold}")
        def capture(module, args, k=key):
            fcin[k] = args[0]
        nets[key].fc.register_forward_pre_hook(capture)
        if nets[key].recoil_projection.bias is not None:
            raise ValueError("Expected a bias-free recoil contribution")
    cfg = config["current"]
    if any(config["previous"].inputs[k] != cfg.inputs[k] for k in INPUTS):
        raise ValueError("Incompatible reconstructed feature ordering")
    features = {name for group in INPUTS for name in cfg.inputs[group]}
    expr = {k: v for k,v in cfg.new_variables.items() if k in features}
    fields = features - expr.keys() | set().union(*(extract_fields_from_expr(v) for v in expr.values()))
    with ExitStack() as stack:
        raw = open_tree(stack, RUN / f"dataset/3fold_{fold}.root")
        original = {}
        for key, path in MODELS.items():
            t = open_tree(stack, path / f"output/fold_{fold}/test/test_scores.root")
            original[key] = t.arrays(["eventNumber", "dsid", "weight", "class_id", "n_jet",
                                      "p_signal", "p_extconv_bkg", "p_recoil_wz"], library="np")
        a, b = original["previous"], original["current"]
        for key in ("eventNumber", "dsid", "weight", "class_id", "n_jet"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Test rows disagree on {key}")
        if len(a["weight"]) != raw.num_entries:
            raise ValueError("Raw and scored event counts differ")
        code = process_codes(a["dsid"], RUN / "dataset/summary.json")
        use = np.isin(code, (-1,0))
        selected = np.flatnonzero(use)
        filled = 0
        out = {m: {mode: np.empty((len(selected), 2), "f4") for mode in MODES} for m in MODELS}
        for start in range(0, len(use), 26000):
            stop = min(start+26000, len(use))
            rows = selected[np.searchsorted(selected, start):np.searchsorted(selected, stop)]
            if not len(rows):
                continue
            d = raw.arrays([*sorted(fields), "dsid", "eventNumber"], entry_start=start, entry_stop=stop,
                           library="ak")[rows-start]
            for k in ("dsid", "eventNumber"):
                if not np.array_equal(ak.to_numpy(d[k]), a[k][rows]):
                    raise ValueError("Different identities for raw and scored rows")
            inputs = raw_inputs(d, cfg, expr)
            for offset in range(0, len(rows), 384):
                size = min(384, len(rows)-offset)
                batch = [x[offset:offset+size] for x in inputs]
                dest = slice(filled, filled+size)
                base_features, recoil = {}, {}
                for m, net in nets.items():
                    logits, aux, _ = net(*batch)
                    out[m]["native"][dest] = logits.softmax(1)[:, (0,2)].numpy()
                    recoil[m] = aux["wz_recoil"].tanh().unsqueeze(1)
                    base_features[m] = fcin[m] - net.recoil_projection(recoil[m])
                for m, other in (("previous", "current"), ("current", "previous")):
                    net = nets[m]
                    off = net.fc(base_features[m])
                    cross = net.fc(base_features[m]+net.recoil_projection(recoil[other]))
                    out[m]["no_recoil"][dest] = off.softmax(1)[:, (0,2)].numpy()
                    out[m]["other_recoil"][dest] = cross.softmax(1)[:, (0,2)].numpy()
                filled += size
        if filled != len(selected):
            raise ValueError("Incomplete application")
    common = {k: a[k][use] for k in ("class_id","n_jet","weight")}
    common["wz"] = code[use] == 0
    reference = {m: np.stack([x["p_signal"][use], x["p_extconv_bkg"][use]], axis=1) for m,x in original.items()}
    return out, reference, common


def analyze(fold):
    out, exported, a = score_events(fold)
    hh, wz = a["class_id"] == 0, a["wz"]
    if not np.array_equal(~hh, wz):
        raise ValueError("Unexpected process in HH/WZ test sample")
    pre = a["n_jet"] >= 1
    w = a["weight"].astype("f8")
    selections, values = {}, []
    for m, cases in out.items():
        cases["exported_native"] = exported[m]
        for mode, p in cases.items():
            if not np.isfinite(p).all():
                raise ValueError("Nonfinite recomputed probabilities")
            cuts = thresholds(p, hh, w, pre)
            selections[m, mode] = []
            for eff, cut in zip(POINTS, cuts):
                accepted = pre & (p[:,1] < .8) & (p[:,0] >= cut)
                selections[m,mode].append(accepted)
                positive_wz = pre & wz & (w > 0)
                values.append(dict(model=m, recoil=mode, target_hh_efficiency=eff, score_cut=float(cut),
                    actual_positive_hh_efficiency=float(w[accepted & hh & (w>0)].sum()/w[pre & hh & (w>0)].sum()),
                    signed_hh=float(w[accepted & hh].sum()), signed_wz=float(w[accepted & wz].sum()),
                    mc_wz_error=float(np.sqrt(np.square(w[accepted & wz]).sum())),
                    positive_wz_rejection=float(w[positive_wz].sum()/w[accepted & positive_wz].sum())))
    deltas = []
    for j, eff in enumerate(POINTS):
        def indicators(m, select):
            return selections[m,select][j][wz].astype("i1")
        vw = w[wz]
        def record(name, delta):
            deltas.append(dict(comparison=name, target_hh_efficiency=eff, wz_yield_change=float(vw @ delta.astype("f8")),
                               paired_wz_mc_error=float(np.linalg.norm(vw*delta)), changed_wz_rows=int((delta!=0).sum())))
        for mode in (*MODES, "exported_native"):
            record("current_minus_previous_"+mode, indicators("current",mode)-indicators("previous",mode))
        gains = {}
        for m in MODELS:
            gains[m] = indicators(m,"no_recoil") - indicators(m,"native")
            record(m+"_wz_removed_by_native_recoil", gains[m])
            record(m+"_substitute_other_recoil", indicators(m,"other_recoil")-indicators(m,"native"))
        record("current_minus_previous_wz_removed_by_recoil", gains["current"]-gains["previous"])
    folder = RUN/f"output/fold_{fold}/evaluation/ancestry_comparison"
    save_table(folder/"all_wz_recoil_yields.csv", values)
    save_table(folder/"all_wz_recoil_paired_changes.csv", deltas)
    checks = {}
    for m in MODELS:
        error = abs(out[m]["native"] - exported[m])
        checks[m] = {"q99_signal_difference":float(np.quantile(error[:,0],.99)),
                     "q99_conversion_difference":float(np.quantile(error[:,1],.99))}
        if max(checks[m].values()) > .005:
            raise ValueError(f"Native application differs from stored scores: {checks[m]}")
    report = dict(fold=fold, test_HH_rows=int(hh.sum()), test_WZ_rows=int(wz.sum()), numerical_checks=checks,
                  normalization="Each case matched to 20%, 30%, 50% of all positive-weight HH with >=1 jet before its conversion veto.",
                  uncertainty="Paired WZ MC at the observed HH cuts. All selected test HH and WZ rows were scored.")
    (folder/"all_wz_recoil_probe.json").write_text(json.dumps(report, indent=2)+"\n")
    plot(values, deltas, folder, fold)
    from crossfold_review import render
    render()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, choices=(0,1), required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    analyze(args.fold)
