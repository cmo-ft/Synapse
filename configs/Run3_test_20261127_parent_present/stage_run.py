"""Stage a verified Synapse run, keeping the checkout and completed snapshots independent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


CONFIG = Path(__file__).resolve().parent
REPO = CONFIG.parents[1]
TRAINING_FILES = ("worker.sh", "launch.py", "evaluate_aux.py", "working_point_composition.py",
                  "data_config.yaml", "model_config.yaml", "run_config.yaml")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_sources():
    src = REPO / "src"
    files = {}
    for module in ("", "core", "loss_functions", "models", "third_party"):
        for path in (src / "synapse" / module).glob("*.py"):
            files[str(path.relative_to(src))] = path
    for name in TRAINING_FILES:
        recipe = "Run3_test_20261127_higgs_tree" if name.endswith(".yaml") else "Run3_test_20261126_ancestry"
        files["recipe/" + name] = CONFIG.parent / recipe / name
    for name in ("plot_background_rejection.py", "process_catalog.py", "trafo60_composition.py", "preprocess.py"):
        files["plots/" + name] = CONFIG.parent / "run3_common" / name
    return files


def stage(run, config, reference=None):
    if reference is None:
        files = checkout_sources()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        provenance = {"checkout_commit": commit, "config": config.name}
    else:
        source = reference / "frozen"
        manifest = json.loads((source / "sha256.json").read_text())
        names = {k for k in manifest if k.startswith(("synapse/", "plots/"))}
        names.update("recipe/" + k for k in TRAINING_FILES)
        files = {name: source / name for name in names}
        for name, path in files.items():
            if digest(path) != manifest[name]:
                raise ValueError(f"Reference checksum failed for {name}")
        provenance = {"reference": reference.name, "reference_manifest": digest(source / "sha256.json")}
    for filename in ("data_config.yaml", "model_config.yaml", "run_config.yaml", "prepare_dataset.py", "hypothesis.py"):
        path = config / filename
        if path.is_file():
            files["recipe/" + filename] = path
    expected = {name: digest(path) for name, path in sorted(files.items())}
    run.mkdir(parents=True, exist_ok=True)
    dest = run / "frozen"
    if dest.exists():
        if json.loads((dest / "sha256.json").read_bytes()) != expected or any(digest(dest/k) != value for k,value in expected.items()):
            raise ValueError(f"A different frozen snapshot exists at {dest}")
    else:
        with TemporaryDirectory(dir=run, prefix=".freeze-") as tmp:
            tree = Path(tmp) / "frozen"
            for name, path in files.items():
                dst = tree / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dst)
            (tree / "sha256.json").write_text(json.dumps(expected, indent=2) + "\n")
            (tree / "revisions.json").write_text(json.dumps(provenance, indent=2) + "\n")
            tree.rename(dest)
    worker = run / "worker.sh"
    if worker.exists() and digest(worker) != expected["recipe/worker.sh"]:
        raise ValueError("Unexpected existing worker")
    if not worker.exists():
        shutil.copyfile(dest / "recipe/worker.sh", worker)
        worker.chmod(0o755)
    diagnostics = run / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for filename in ("analyze.py", "track.py", "workflow.sh"):
        src = Path(__file__).resolve().parent / filename
        dst = diagnostics / filename
        if dst.exists() and digest(dst) != digest(src):
            raise ValueError(f"Conflicting existing diagnostic {dst}")
        if not dst.exists():
            shutil.copyfile(src, dst)
    print(f"Verified {len(expected)} frozen sources at {dest}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG, help="Recipe overrides (default: parent_present)")
    p.add_argument("--reference", type=Path, help="Use a completed frozen run as the source instead of this checkout")
    args = p.parse_args()
    stage(args.run.resolve(), args.config.resolve(), args.reference.resolve() if args.reference else None)
