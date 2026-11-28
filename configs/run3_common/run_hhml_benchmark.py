"""Run the native hhml_eval score and Trafo60 benchmark separately on each scored fold."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import uproot
import yaml

from trafo60_edges import transformtool_source
from fold_recipe import fold_roles

EVALUATOR = Path(__file__).resolve().parents[2] / "hhml_eval"
sys.path.insert(0, str(EVALUATOR))
from convert_root_to_h5 import root_to_compound_h5


def benchmark_config(scores, fold, source):
    fold_name = f"fold_{fold}"
    cfg = yaml.safe_load((EVALUATOR / "eval_config.yaml").read_text())
    syn = "(p_extconv_bkg < 0.8) & (n_jet >= 1)"
    eve = "(evenet_valid != 0) & (evenet_pConv < 0.8) & (n_jet >= 1)"
    classifiers = {}
    for name, score, selection, color, line in (
        ("Synapse", "p_signal", syn, "#20639B", "-"),
        ("EveNet", "evenet_pS", eve, "#C45924", "-"),
        ("XGB_SynapseSR", "xgb_score", syn, "#27836B", "--"),
        ("XGB_EveNetSR", "xgb_score", eve, "#7346A1", ":"),
    ):
        veto = ["evenet_pConv", "evenet_valid"] if selection == eve else ["p_extconv_bkg"]
        classifiers[name] = dict(columns={key: key for key in [score, "n_jet", *veto]}, expression=score,
                                 selection=selection, split_train_test_val=False, score_plot_color=color,
                                 score_plot_config={"test": dict(type="hist", histtype="step", linewidth=2, linestyle=line)})
    data = dict(file_pattern="scores.h5", file_block="tree", weight_col="weight", label_col="label_signal",
                base_bin_num=20000, base_dir=".", fold_dirs={fold_name: "."}, model_name=fold_name, classifiers=classifiers)
    roc = cfg["roc"]
    roc.update(reference_classifier="Synapse", reference_fold=fold_name, roc_range=[0.05, 1.0])
    dist = cfg["score_distribution"]
    dist.update(x_min=0.0, x_max=1.0, bin_num=20, baseline_classifier="Synapse", fold_reference=None,
                fold_colors={fold_name: "#20639B"})
    dist["trafo"].update(use_hhard_binning=False, trafo_six_y=10, trafo_six_z=5,
                         trafo_sixty_mc_low_bound=3.0, trafo_six_mc_stat_up_bound=0.2)
    return dict(input_data=[data], roc=roc, score_distribution=dist, include_roc=True, include_roc_inverse=True,
                include_score_distribution=True, include_score_distribution_trafo=True, source_root=str(scores),
                event_number_remainder=fold, transformtool_source=str(source))


def run_fold(scores, fold, folder, runtime, output, source):
    work = runtime / folder
    work.mkdir()
    fold_name = f"fold_{fold}"
    with uproot.open(scores) as f:
        events = f["tree"]["eventNumber"].array(library="np")
        if not len(events) or not np.all(events % 3 == fold):
            raise ValueError(f"Incorrect event-number remainder in {scores}")
    branches = ["eventNumber", "dsid", "weight", "label_signal", "n_jet", "p_signal", "p_extconv_bkg",
                "evenet_pS", "evenet_pConv", "evenet_valid", "xgb_score"]
    root_to_compound_h5(str(scores), "tree", str(work/"scores.h5"), branches)
    config = benchmark_config(scores, fold, source)
    (work/"benchmark.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime/"hhml_eval") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(runtime/"matplotlib"), PYTHONDONTWRITEBYTECODE="1")
    with (work/"native.log").open("w") as log:
        result = subprocess.run([sys.executable, "-B", "-m", "eval", "benchmark.yaml", "--eval_fold", fold_name,
                                 "--out_dir", "."], cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("\n".join((work/"native.log").read_text(errors="replace").splitlines()[-35:]))
    sig = json.loads((work/"sig_with_trafo.json").read_text())
    print(folder, {model: values[fold_name]["test"] for model, values in sig.items()}, flush=True)

    dest = output / folder
    dest.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest) as publish:
        staged = Path(publish)
        shutil.copytree(work, staged, dirs_exist_ok=True)
        files = sorted((p for p in staged.rglob("*") if p.is_file()), key=lambda p: p.name == "sig_with_trafo.json")
        for path in files:
            target = dest / path.relative_to(staged)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)


def write_summary(output, held_out=0):
    report = {"evaluator": str(EVALUATOR/"eval.py"), "binning": "Trafo60 edges fitted and evaluated on the same fold",
              "yield_weights": "signed nominal MC weights", "uncertainty": "background MC statistical uncertainty",
              "folds": {}}
    roles = fold_roles(held_out)
    for role, index in (("held_out", held_out), ("validation", roles["validation"])):
        fold = f"fold_{index}"
        folder = output / role
        fixed = json.loads((folder/"sig_without_trafo.json").read_text())
        adaptive = json.loads((folder/"sig_with_trafo.json").read_text())
        results = {model: dict(fixed20_z=fixed[model][fold]["test"], trafo60_z=values[fold]["test"],
                               trafo60_bins=len(values[fold]["test_signal_hists"])) for model, values in adaptive.items()}
        report["folds"][fold] = {"role": role, "models": results}
    with TemporaryDirectory(dir=output) as temp:
        staged = Path(temp) / "summary.json"
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.replace(output/"summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0, help="Held-out event-number remainder")
    args = parser.parse_args()
    source = transformtool_source()
    uproot.open.defaults["handler"] = uproot.source.file.MemmapSource
    with TemporaryDirectory(prefix="synapse-hhml-eval-") as tmp:
        runtime = Path(tmp)
        # The native module loads and compiles HistoTransform beside its own checkout.
        (runtime/"hhml_eval").symlink_to(EVALUATOR, target_is_directory=True)
        for rel in ("Root/HistoTransform.cxx", "TransformTool/HistoTransform.h"):
            target = runtime / "TransformTool" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/rel, target)
        roles = fold_roles(args.fold)
        run_fold(args.test.resolve(), args.fold, "held_out", runtime, args.out.resolve(), source)
        run_fold(args.validation.resolve(), roles["validation"], "validation", runtime, args.out.resolve(), source)
        write_summary(args.out.resolve(), args.fold)


if __name__ == "__main__":
    main()
