"""Freeze the exact Python sources and recipes for the Run 3 ancestry workers."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from paths import RECIPE, REPO, RUN


def main():
    root=RUN/"frozen"
    RUN.mkdir(parents=True,exist_ok=True)
    if root.exists():
        for name,digest in json.loads((root/"sha256.json").read_text()).items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f"Changed frozen source: {name}")
        if not (RUN/"worker.sh").is_file():
            shutil.copyfile(root/"recipe/worker.sh",RUN/"worker.sh")
        return
    with TemporaryDirectory(dir=RUN,prefix=".freeze-") as tmp:
        stage=Path(tmp)/"frozen"
        files=[]
        for sub in ("","core","loss_functions","models","third_party"):
            for src in (REPO/"src/synapse"/sub).glob("*.py"):
                files.append((src,stage/src.relative_to(REPO/"src")))
        for pattern in ("*.py","*.yaml","*.hxx","*.sh"):
            for src in RECIPE.glob(pattern):
                files.append((src,stage/"recipe"/src.name))
        for src,dest in files:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest)
        helpers=stage/"plots"
        helpers.mkdir()
        for name in ("plot_background_rejection.py","trafo60_composition.py","process_catalog.py","preprocess.py"):
            path=helpers/name
            shutil.copyfile(REPO/"configs/run3_common"/name, path)
            files.append((None,path))
        hashes={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for _,p in files}
        (stage/"sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
        commits={"HEAD":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
        (stage/"revisions.json").write_text(json.dumps(commits,indent=2)+"\n")
        stage.rename(root)
    shutil.copyfile(root/"recipe/worker.sh",RUN/"worker.sh")
    print(root,flush=True)


if __name__=="__main__":
    main()
