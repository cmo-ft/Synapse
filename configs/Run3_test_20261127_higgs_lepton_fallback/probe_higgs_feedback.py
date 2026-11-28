"""Disable the Higgs matrix or the WZ recoil path separately in two frozen checkpoints."""
import argparse
from contextlib import ExitStack
import csv
import json
import sys

import awkward as ak
import numpy as np
import torch
import uproot

from paths import ANCESTRY_RUN, RUN

sys.path.insert(0, str(RUN / "frozen"))
sys.path.insert(0, str(RUN / "frozen/plots"))
from synapse.core.config import DataConfig, ModelConfig
from synapse.core.tools import build_new_variables, extract_fields_from_expr
from synapse.models.ParticleTransformer import ParticleTransformerSimple
from process_catalog import GROUPS, process_codes

MODELS = {"previous": ANCESTRY_RUN, "current": RUN}
INPUTS = ("obj_feats", "evt_feats", "obj_4vecs", "mask")
BOUNDARY = .945
PRESCALE = 60


def open_tree(stack, path):
    return stack.enter_context(uproot.open(path, handler=uproot.source.file.MemmapSource, array_cache=None))["tree"]


def read_checkpoint(folder, checkpoint=None):
    cfg = ModelConfig(str(folder / "test/model_config.yaml"))
    data = DataConfig(str(folder / "test/data_config.yaml"))
    channel = list(cfg.shared_parent_outputs).index("relationHiggs")
    ckpts = [checkpoint] if checkpoint is not None else list((folder/"fit/checkpoints").glob("BEST-*.ckpt"))
    if len(ckpts) != 1 or not ckpts[0].is_file():
        raise ValueError(f"Expected one selected checkpoint in {folder}")
    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)["state_dict"]
    model = ParticleTransformerSimple(**cfg.model_params_for(data))
    model.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}, strict=True)
    model.eval()
    if not model.has_input_norm:
        raise ValueError("Checkpoint lacks frozen input normalization")
    return data, model, channel, ckpts[0].name


def raw_inputs(raw, data, expr):
    expanded = build_new_variables(raw, expr, supress_warning=True)
    result = []
    for group in INPUTS:
        values = [ak.to_numpy(expanded[key]) for key in data.inputs[group]]
        x = np.stack(values, axis=1).astype("f4")
        x[np.isnan(x)] = 0
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite input in {group}")
        result.append(torch.from_numpy(x))
    return result


def assign(data, ids, index, prefix, arr):
    for col, name in ((0, "score"), (2, "conv")):
        data[prefix + "_" + name][ids[index]] = arr[:, col].cpu().numpy()


def cut_at_20(score, conv, hh, weight, pre, total_positive_hh):
    candidates = np.flatnonzero(pre & hh & (weight > 0) & (conv < .8))
    order = candidates[np.argsort(-score[candidates], kind="stable")]
    return float(score[order[np.searchsorted(weight[order].cumsum(), .2*total_positive_hh)]])


@torch.inference_mode()
def evaluate(fold, output):
    models, axes, records, configs, fc_input = {}, {}, {}, {}, {}
    for key, run in MODELS.items():
        configs[key], models[key], axes[key], ckpt = read_checkpoint(run / f"output/fold_{fold}")
        head = models[key].shared_parent_head
        records[key] = {"checkpoint": ckpt, "feedback_coefficients_before": head.feedback.weight.tolist(),
                        "higgs_channel": axes[key]}
        # The frozen classifier receives the HH event features plus a bias-free WZ recoil projection.
        # Capture its input so the same event can be evaluated with the recoil contribution removed.
        def capture(module, inputs, name=key):
            fc_input[name] = inputs[0]
        models[key].fc.register_forward_pre_hook(capture)
    data = configs["current"]
    if any(configs["previous"].inputs[k] != data.inputs[k] for k in INPUTS):
        raise ValueError("Old and current reconstructed input order differs")
    features = {v for group in INPUTS for v in data.inputs[group]}
    expr = {k: v for k, v in data.new_variables.items() if k in features}
    if any(configs["previous"].new_variables.get(k) != v for k, v in expr.items()):
        raise ValueError("Reconstructed variable definitions differ")
    fields = (features - expr.keys()) | set().union(*(extract_fields_from_expr(v) for v in expr.values()))
    with ExitStack() as stack:
        raw = open_tree(stack, RUN / f"dataset/3fold_{fold}.root")
        scores = {}
        for m, run in MODELS.items():
            t = open_tree(stack, run / f"output/fold_{fold}/test/test_scores.root")
            scores[m] = t.arrays(["eventNumber", "dsid", "weight", "class_id", "n_jet",
                                  "p_signal", "p_extconv_bkg"], library="np")
        full = scores["previous"]
        for key in ("eventNumber", "dsid", "weight", "class_id", "n_jet"):
            if not np.array_equal(full[key], scores["current"][key]):
                raise ValueError(f"Different test {key}")
        if raw.num_entries != len(full["weight"]):
            raise ValueError("Dataset row count differs")
        # Include the full nominal tail and a simple random sample of the remainder.
        high = np.maximum(full["p_signal"], scores["current"]["p_signal"]) >= BOUNDARY
        rand = np.random.default_rng(40409 + fold)
        keep = high | (rand.random(len(high)) < 1 / PRESCALE)
        ids = np.cumsum(keep, dtype="i8")-1
        out = {m: {k+n: np.full(int(keep.sum()), np.nan, dtype="f4")
                   for k in ("nominal_", "ablated_", "no_recoil_") for n in ("score", "conv")} for m in MODELS}
        for start in range(0, len(keep), 22000):
            stop = min(start+22000, len(keep))
            rows = np.flatnonzero(keep[start:stop]) + start
            if not len(rows):
                continue
            chunk = raw.arrays([*sorted(fields), "eventNumber", "dsid"],
                               entry_start=start, entry_stop=stop, library="ak")[rows-start]
            for key in ("dsid", "eventNumber"):
                if not np.array_equal(ak.to_numpy(chunk[key]), full[key][rows]):
                    raise ValueError("Input and exported score rows disagree")
            inputs = raw_inputs(chunk, data, expr)
            for offset in range(0, len(rows), 384):
                index = slice(offset, min(offset + 384, len(rows)))
                batch = [x[index] for x in inputs]
                for m, net in models.items():
                    head = net.shared_parent_head.feedback.weight
                    base = head[:, axes[m]].clone()
                    logits, aux, _ = net(*batch)
                    nominal = logits.softmax(1)
                    assign(out[m], ids[rows], index, "nominal", nominal)
                    contribution = net.recoil_projection(aux["wz_recoil"].tanh().unsqueeze(1))
                    no_recoil = net.fc(fc_input[m] - contribution).softmax(1)
                    assign(out[m], ids[rows], index, "no_recoil", no_recoil)
                    try:
                        head[:, axes[m]].zero_()
                        ablated = net(*batch)[0].softmax(1)
                        assign(out[m], ids[rows], index, "ablated", ablated)
                    finally:
                        head[:, axes[m]].copy_(base)
        npre = full["n_jet"][keep] >= 1
        h = full["class_id"][keep] == 0
        groups = process_codes(full["dsid"][keep], RUN / "dataset/summary.json")
        nominal_weight = full["weight"][keep].astype("f8")
        w = nominal_weight * np.where(high[keep], 1, PRESCALE)
        cuts = json.loads((output/"event_comparison.json").read_text())["validation"]["split"]
        total_positive_hh = full["weight"][(full["n_jet"] >= 1) & (full["class_id"] == 0) & (full["weight"] > 0)].sum(dtype="f8")
        tables = {"ablated": [], "no_recoil": []}
        for m, changed in out.items():
            old = changed["nominal_score"]
            if not all(np.isfinite(v).all() for v in changed.values()):
                raise ValueError("Missing recomputed scores")
            delta = np.abs(old - scores[m]["p_signal"][keep])
            delta_conv = np.abs(changed["nominal_conv"] - scores[m]["p_extconv_bkg"][keep])
            records[m]["absolute_nominal_gpu_cpu_score_difference"] = dict(
                median=float(np.median(delta)), q99=float(np.quantile(delta, .99)), max=float(delta.max()),
                conv_q99=float(np.quantile(delta_conv, .99)))
            if np.quantile(delta, .99) > .005 or np.quantile(delta_conv, .99) > .005:
                raise ValueError(f"CPU nominal score fails to reproduce the exported GPU score: {records[m]}")
            cut = cuts[m+"_score_"+m+"_veto"]["cuts"][0]
            records[m]["fixed_validation_cut"] = cut
            previous = npre & (changed["nominal_conv"] < .8) & (old >= cut)
            official = npre & (scores[m]["p_extconv_bkg"][keep] < .8) & (scores[m]["p_signal"][keep] >= cut)
            records[m]["nominal_cpu_vs_exported_threshold_disagreements"] = int((previous != official).sum())
            cut_base = cut_at_20(old, changed["nominal_conv"], h, w, npre, total_positive_hh)
            for branch in tables:
                ab, av = changed[branch+"_score"], changed[branch+"_conv"]
                cut_off = cut_at_20(ab, av, h, w, npre, total_positive_hh)
                for mode, before, after in (("fixed_validation", cut, cut), ("matched_test", cut_base, cut_off)):
                    a0 = npre & (changed["nominal_conv"] < .8) & (old >= before)
                    b0 = npre & (av < .8) & (ab >= after)
                    for code, process in [(-2, "all backgrounds"), (-1, "HH"), *enumerate(GROUPS.values())]:
                        selected = ~h if code == -2 else groups == code
                        a, b = (s & selected for s in (a0, b0))
                        shift = b.astype("i1") - a.astype("i1")
                        wt = np.abs(w[selected])
                        low = ~high[keep]
                        # Bernoulli sampling variance of the yield difference outside the complete tail.
                        sample_var = np.sum((1 - 1/PRESCALE) * (w[low] * shift[low])**2)
                        tables[branch].append(dict(model=m, working_point=mode, process=process,
                            nominal_cut=before, ablated_cut=after, nominal=float(w[a].sum()), ablated=float(w[b].sum()),
                            delta=float((w*shift).sum()), low_stratum_sample_error=float(np.sqrt(sample_var)),
                            changed_loose_sample_rows=int((low & (shift != 0)).sum()),
                            changed_full_tail_rows=int((~low & (shift != 0)).sum()),
                            abs_weight_mean_score_shift=float(np.average(ab[selected]-old[selected], weights=wt)),
                            abs_weight_mean_absolute_score_shift=float(np.average(abs(ab[selected]-old[selected]), weights=wt))))
    result = dict(fold=fold, seed=40409+fold, sampled_rows=int(keep.sum()), full_tail_rows=int(high.sum()),
                  lower_score_rows_sampled=int((keep & ~high).sum()), prob_below_boundary=1/PRESCALE,
                  full_tail_score_boundary=BOUNDARY, models=records)
    (output / "higgs_feedback_probe.json").write_text(json.dumps(result, indent=2)+"\n")
    for branch, name in (("ablated", "higgs_feedback"), ("no_recoil", "wz_recoil")):
        with (output / (name + "_yields.csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tables[branch][0])
            writer.writeheader()
            writer.writerows(tables[branch])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", type=int, required=True, choices=(0, 1))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = RUN / f"output/fold_{args.fold}/evaluation/ancestry_comparison"
    evaluate(args.fold, folder)
