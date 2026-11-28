"""Freeze the verified tree-only training code and the new dataset recipe."""
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from paths import RECIPE, RUN, TREE_RUN


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = TREE_RUN / "frozen"
    manifest = json.loads((old / "sha256.json").read_bytes())
    training = {"recipe/" + x for x in ("worker.sh", "launch.py", "evaluate_aux.py", "working_point_composition.py",
                                        "run_config.yaml", "model_config.yaml", "data_config.yaml")}
    sources = {k: old / k for k in manifest if k.startswith(("synapse/", "plots/")) or k in training}
    if not training <= sources.keys() or any(digest(p) != manifest[k] for k, p in sources.items()):
        raise ValueError("Reference training snapshot failed verification")
    for k in ("paths.py", "mask_sources.py", "higgs_labels.py", "publish.py", "verify_dataset.py", "check_run.py",
              "stage_run.py"):
        sources["recipe/" + k] = RECIPE / k
    expected = {k: digest(v) for k, v in sources.items()}
    final = RUN / "frozen"
    if final.exists():
        if json.loads((final / "sha256.json").read_bytes()) != expected or any(digest(final/k) != v for k,v in expected.items()):
            raise ValueError("Run snapshot differs from the recipe")
    else:
        with TemporaryDirectory(dir=RUN, prefix=".freeze-") as tmp:
            stage = Path(tmp) / "frozen"
            for name, source in sources.items():
                dst = stage / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, dst)
            (stage / "sha256.json").write_text(json.dumps(expected, indent=2) + "\n")
            (stage / "revisions.json").write_text(json.dumps(dict(reference=TREE_RUN.name,
                  reference_manifest_sha256=digest(old/"sha256.json")), indent=2) + "\n")
            stage.rename(final)
    worker = RUN / "worker.sh"
    if worker.exists() and digest(worker) != expected["recipe/worker.sh"]:
        raise ValueError(f"An unrelated worker already exists at {worker}")
    if not worker.exists():
        shutil.copyfile(final / "recipe/worker.sh", worker)
    worker.chmod(0o755)
    desc = (TREE_RUN / "fold1.sub").read_text()
    required = (str(TREE_RUN), f"batch_name = {TREE_RUN.name}_fold1")
    if not all(s in desc for s in required):
        raise ValueError("Unexpected reference submission description")
    desc = desc.replace(TREE_RUN.name, RUN.name)
    path = RUN / "fold1.sub"
    if path.exists() and path.read_text() != desc:
        raise ValueError(f"An unrelated submission description already exists at {path}")
    if not path.exists():
        path.write_text(desc)
    print(f"Frozen {len(expected)} files under {final}")


if __name__ == "__main__":
    main()
