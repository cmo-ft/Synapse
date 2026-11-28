"""Snapshot the Higgs decay-tree recipe with the ancestry run's model and evaluation sources."""
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from paths import BASELINE, RECIPE, REPO, RUN


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous=BASELINE/"frozen"
    files={}
    for name in json.loads((previous/"sha256.json").read_text()):
        if name.startswith(("synapse/","plots/")):
            files[name]=previous/name
    files["synapse/loss_functions/lepton_objects.py"]=REPO/"src/synapse/loss_functions/lepton_objects.py"
    old=REPO/"configs"/BASELINE.name
    for name in ("worker.sh","evaluate_aux.py","working_point_composition.py"):
        files["recipe/"+name]=old/name
    files["recipe/launch.py"]=previous/"recipe/launch.py"
    for name in ("data_config.yaml","model_config.yaml","run_config.yaml","paths.py","higgs_decay.hxx",
                 "prepare_raw.py","raw_worker.sh","publish.py","submit_raw.py","check_run.py"):
        files["recipe/"+name]=RECIPE/name
    hashes={name:digest(src) for name,src in files.items()}
    final=RUN/"frozen"
    if final.exists():
        recorded=json.loads((final/"sha256.json").read_text())
        if recorded!=hashes or any(digest(final/name)!=h for name,h in recorded.items()):
            raise RuntimeError("Frozen snapshot differs from source")
    else:
        with TemporaryDirectory(dir=RUN,prefix=".freeze-") as tmp:
            stage=Path(tmp)/"frozen"
            for name,source in files.items():
                path=stage/name
                path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,path)
            (stage/"sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
            (stage/"revisions.json").write_text(json.dumps({"baseline":str(BASELINE),
                "baseline_manifest_sha256":digest(previous/"sha256.json")},indent=2)+"\n")
            stage.rename(final)
    worker=RUN/"worker.sh"
    if not worker.exists():
        shutil.copyfile(final/"recipe/worker.sh",worker)
    print(final)


if __name__=="__main__":
    main()
