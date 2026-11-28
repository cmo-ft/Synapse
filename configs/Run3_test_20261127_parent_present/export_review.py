"""Export the completed parent_present fold 1 for review on another server."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from shutil import copyfile
import subprocess
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from review_examples import export_examples


REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "run_dir/Run3_test_20261127_parent_present"
PREVIOUS = "Run3_test_20261127_higgs_lepton_fallback"
BASELINE = "0073327"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *map(str, args)])


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, target)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_history(run, fit, dest):
    dest.mkdir()
    rows = [json.loads(f.read_text()) for f in sorted((fit / "epochs").glob("epoch_*.json"))]
    if [r["epoch"] for r in rows] != list(range(1, 51)):
        raise ValueError("Expected the 50 completed fold 1 epochs")
    best = max(rows, key=lambda r: r["val_Significance20_30_50_epoch"])
    with (dest / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    copy(run / "epochs.csv", dest / "history_vs_previous.csv")
    comparison = read_csv(dest / "history_vs_previous.csv")
    epochs = [int(r["epoch"]) for r in comparison]
    if epochs != [r["epoch"] for r in rows]:
        raise ValueError("Mismatched epoch numbers in comparison history")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, [("loss", "Event loss"), ("auc", "Macro AUC"),
                                             ("r20", "Background rejection at 20% HH efficiency"),
                                             ("z", "Three-bin expected significance")]):
        ax.plot(epochs, [float(r[key]) for r in comparison], color="#236c92", label="Parent present")
        ax.plot(epochs, [float(r["reference_" + key]) for r in comparison],
                color="#795493", linestyle="--", label="Previous NN")
        ax.axvline(best["epoch"], color="0.5", lw=1, ls=":", label=f"Selected epoch {best['epoch']}")
        ax.set(title=title, xlabel="Epoch", xlim=(1, 50))
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fold 1 validation")
    fig.savefig(dest / "validation_history.pdf")
    plt.close(fig)
    return best


def export_sources(run, dest):
    frozen = run / "frozen"
    manifest = json.loads((frozen / "sha256.json").read_text())
    reviewed = dest / "frozen"
    for name, expected in manifest.items():
        copy(frozen / name, reviewed / name)
        if sha(reviewed / name) != expected:
            raise ValueError(f"Archived run source checksum mismatch: {name}")
    for name in ("sha256.json", "revisions.json"):
        copy(frozen / name, reviewed / name)
    commit = git(REPO, "rev-parse", "HEAD").decode().strip()
    branch = git(REPO, "symbolic-ref", "--short", "HEAD").decode().strip()
    base = git(REPO, "rev-parse", BASELINE).decode().strip()
    git(REPO, "bundle", "create", dest / "Synapse.bundle", f"refs/heads/{branch}", "HEAD")
    (dest / "preprocessing.md").write_bytes(
        git(REPO, "show", f"{commit}:configs/Run3_test_20261127_parent_present/preprocessing.md")
    )
    (dest / "changes.patch").write_bytes(git(REPO, "diff", "--no-ext-diff", "--binary", base, commit))
    evaluator = git(REPO, "rev-parse", commit + ":hhml_eval").decode().strip()
    transform = git(REPO / "TransformTool", "rev-parse", "HEAD").decode().strip()
    for name, revision in (("hhml_eval", evaluator), ("TransformTool", transform)):
        git(REPO / name, "archive", "--format=tar.gz", "-o", dest / (name + ".tar.gz"), revision)
    # Native Trafo60 reads these two files directly from the source checkout.
    for name in ("Root/HistoTransform.cxx", "TransformTool/HistoTransform.h"):
        if git(REPO / "TransformTool", "show", f"{transform}:{name}") != (REPO / "TransformTool" / name).read_bytes():
            raise ValueError(f"Native TransformTool source differs from its archived revision: {name}")
    differences = [name for name in manifest if name.startswith("synapse/") and
                   git(REPO, "show", f"{commit}:src/{name}") != (reviewed / name).read_bytes()]
    packages = ("torch", "lightning", "numpy", "awkward", "uproot", "h5py", "pandas")
    return dict(branch=branch, commit=commit, baseline=base, hhml_eval=evaluator, TransformTool=transform,
                frozen_source_files=len(manifest), frozen_synapse_differences=differences,
                export_environment={p: version(p) for p in packages})


def export_split(run, root, split):
    dest = root / split
    evaluation = run / "output/fold_1/evaluation"
    comparison = evaluation / ("comparison_" + PREVIOUS) / split
    for name in ("rejection_by_background.pdf", "working_points.csv", "roc_curves.csv", "vvv_working_points.csv"):
        copy(comparison / name, dest / name)
    copy(evaluation / split / "auxiliary.json", dest / "auxiliary.json")
    for stem in ("composition_at_hh_working_points", "trafo60_last_three_bins"):
        for extension in (".csv", ".pdf"):
            name = stem + extension
            copy(evaluation / split / "background_rejection" / name, dest / name)
    role, remainder = ("validation", 2) if split == "validation" else ("held_out", 1)
    native = evaluation / "hhml_eval" / role
    for name in ("scores.h5", "benchmark.yaml", "sig_with_trafo.json", "sig_without_trafo.json"):
        copy(native / name, dest / "hhml_eval" / name)
    for name in ("comparison.pdf", "comparison_trafo.pdf", "roc.pdf", "roc_inverse.pdf"):
        copy(native / f"fold_{remainder}" / name, dest / "hhml_eval" / name)


def review_text(source, best, results):
    models = {"New Synapse": "Parent present", "Previous Synapse": "Higgs lepton fallback",
              "EveNet": "EveNet", "XGBoost": "XGBoost"}
    lines = [
        "# Parent-present ParT · Run 3, fold 1",
        "",
        f"Epoch {best['epoch']} of 50 was selected by maximum validation three-bin significance. "
        "The same checkpoint scored validation (2,297,211 MC events) and test (2,295,545). "
        "Training used 2,296,499 SR + ExtConvCR events and 518,851 WZCR events. "
        "The WZCR events supervised the Z-pair and WZ-recoil tasks.",
        "",
        "## Performance",
        "",
        "| Split | Model | Rejection at 20% HH | Three-bin Z | Trafo60 Z |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results:
        if row["cuts"] == "matched" and row["model"] in models:
            lines.append(f"| {row['split']} | {models[row['model']]} | {float(row['r20']):.1f} | "
                         f"{float(row['z_20_30_50']):.5f} | {float(row['trafo60']):.5f} |")
    lines += [
        "",
        "The displayed 20% HH score threshold is fitted separately on each split. Rejection uses positive MC weights "
        "and the jet-preselection denominator before the conversion veto. Three-bin Z combines the 0–20%, "
        "20–30%, and 30–50% HH bands with signed yields and background MC variance. The native Trafo60 benchmark "
        "fits its binning on each evaluated split. The results CSV also includes test numbers with validation thresholds.",
        "",
        "## Where to look",
        "",
        "- Start with `validation/rejection_by_background.pdf` and `test/rejection_by_background.pdf`. "
        "They compare parent_present, the preceding NN, EveNet and XGBoost at matched HH efficiency. "
        "The two pages show the within-region and common-preselection conventions.",
        "- Each split includes the corresponding working points, ROC curve coordinates, VVV counts, "
        "background composition at 20/30/50% HH efficiency, and the final three Trafo60 bins. "
        "`auxiliary.json` reports the object and pair-task evaluation.",
        "- `comparison/results.csv` and `comparison/signed_yield_at_20.csv` distinguish matched score cuts "
        "from cuts fitted on validation and then applied to test.",
        "- `training/validation_history.pdf` compares all 50 epochs with the preceding NN. "
        "`training/history.csv` contains original epoch metrics and optimizer steps. "
        "`model/best.ckpt` is the selected Lightning checkpoint. Its original name and zero-based index are "
        "recorded in `provenance/source.json`. The resolved training YAMLs are next to the checkpoint.",
        "- `validation/hhml_eval/scores.h5` and `test/hhml_eval/scores.h5` are the native, per-event benchmark "
        "tables with event and dataset IDs, signed weight, HH label, jet count, Synapse HH and conversion scores, "
        "EveNet scores/validity, and XGBoost score. They support fresh event-level checks. The full object-score "
        "ROOT trees and raw training ntuples remain in the original run on EOS.",
        "- `provenance/dataset_audit.json` contains the pair counts and per-branch hashes for each labelled fold. "
        "`dataset_summary.json` in that directory contains class yields and the DSID process catalogue. "
        "Absolute paths in archived YAML or provenance files identify the CERN inputs.",
        "- `examples/` contains corresponding fold inputs, object truth, and the full original score branches. "
        "Each split uses short windows around examples of HH, each background process and the conversion class. "
        "`examples/entries.csv` identifies the original rows. Use the full split scores for efficiencies and yields. "
        "With the Synapse Python environment active, `python -B examples/check_checkpoint.py .` applies the "
        "checkpoint on CPU and checks every event, object, recoil and shared-parent score against its saved value.",
        "",
        "## Review the code offline",
        "",
        f"The Git bundle contains branch `{source['branch']}` at `{source['commit']}`. "
        f"The preceding organized commit is `{source['baseline'][:7]}`. "
        "Open `source/changes.patch` to read the complete diff directly, or clone the bundle locally.",
        "",
        "```bash",
        f"git clone -b {source['branch']} source/Synapse.bundle Synapse",
        f"git -C Synapse log --oneline {source['baseline'][:7]}..HEAD",
        f"git -C Synapse diff --stat {source['baseline'][:7]}..HEAD",
        "```",
        "",
        "Read `Synapse/configs/Run3_test_20261127_parent_present/README.md` for changes in the model, labels and training. "
        "The commands to rebuild the six MC folds are in `source/preprocessing.md`. "
        "The detailed truth definition and open questions are in "
        "`Synapse/configs/Run3_test_20261127_higgs_lepton_fallback/truth_labels.md`. "
        "`source/frozen/` contains the exact training snapshot. Its `sha256.json` documents the historical file bytes. "
        "The native dependencies can also be unpacked offline.",
        "",
        "```bash",
        "mkdir -p Synapse/hhml_eval Synapse/TransformTool",
        "tar -xzf source/hhml_eval.tar.gz -C Synapse/hhml_eval",
        "tar -xzf source/TransformTool.tar.gz -C Synapse/TransformTool",
        "```",
        "",
        "The tables and PDFs can be reviewed directly. Loading the checkpoint requires the Synapse/PyTorch "
        "environment. Running new applications also requires reconstructed event inputs in Synapse fold format. "
        "To verify every file after transfer, run `sha256sum -c SHA256SUMS` from the package directory.",
    ]
    return "\n".join(lines) + "\n"


def export(run, out):
    if out.exists():
        raise FileExistsError(f"Choose a new export directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fit = run / "output/fold_1/fit"
    evaluation = fit.parent / "evaluation"
    for status in ("exit_status.txt", "pipeline.exit"):
        if (fit.parent / status).read_text().strip() != "0":
            raise ValueError(f"Incomplete training/evaluation: {status}")
    with TemporaryDirectory(dir=out.parent, prefix=".parent-present-export-") as temp:
        root = Path(temp) / out.name
        root.mkdir()
        best = write_history(run, fit, root / "training")
        checkpoint, = (fit / "checkpoints").glob("BEST-*.ckpt")
        saved_epoch = int(re.search(r"epoch=(?:epoch=)?(\d+)", checkpoint.name).group(1))
        if saved_epoch + 1 != best["epoch"]:
            raise ValueError("Selected checkpoint disagrees with epoch history")
        copy(checkpoint, root / "model/best.ckpt")
        for name in ("model_config.yaml", "data_config.yaml", "run_config.yaml"):
            copy(fit / name, root / "model" / name)
        for split in ("validation", "test"):
            export_split(run, root, split)
        comparison = evaluation / ("comparison_" + PREVIOUS)
        for name in ("comparison.md", "results.csv", "signed_yield_at_20.csv"):
            copy(comparison / name, root / "comparison" / name)
        copy(evaluation / "hhml_eval/summary.json", root / "comparison/hhml_eval_summary.json")
        for name in ("audit", "summary"):
            copy(run / "dataset" / (name + ".json"), root / "provenance" / ("dataset_" + name + ".json"))
        export_examples(run, root)
        source = export_sources(run, root / "source")
        source.update(checkpoint_source=checkpoint.name, checkpoint_epoch_index=saved_epoch,
                      selected_epoch=best["epoch"], source_run=str(run), fold=1)
        (root / "provenance/source.json").write_text(json.dumps(source, indent=2) + "\n")
        (root / "README.md").write_text(review_text(source, best, read_csv(comparison / "results.csv")))
        files = sorted(p for p in root.rglob("*") if p.is_file())
        (root / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in files))
        root.rename(out)
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.run.resolve(), args.out.resolve())
