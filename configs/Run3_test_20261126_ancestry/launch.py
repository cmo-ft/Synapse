"""Resolve one fold and run training or score a specified best checkpoint."""
import argparse
import json
from pathlib import Path
import yaml
import uproot


def configs(recipe, dataset, workspace, fold, stage, checkpoint=None):
    def load(name):
        return yaml.safe_load((recipe/(name+"_config.yaml")).read_text())
    data,model,run=map(load,("data","model","run"))
    train,val=(fold+2)%3,(fold+1)%3
    data["train_files"]=[str(dataset/f"3fold_{train}.root"),str(dataset/f"wzcr_{train}.root")]
    data["val_files"]=[str(dataset/f"3fold_{val}.root")]
    data["test_files"]=[str(dataset/f"3fold_{val if stage=='validation' else fold}.root")]
    sow=json.loads((dataset/"summary.json").read_text())["sum_abs_weights_by_remainder"][train]
    data["weights"]["weight_balance_factors"]=[sow[1]/w for w in sow]
    out=workspace/("fit" if stage=="train" else stage)
    run["run_dir"]=str(out)
    if stage!="train":
        if not checkpoint or not checkpoint.is_file():
            raise ValueError("Scoring needs a completed checkpoint")
        run["load_model"]=str(checkpoint)
        data["inputs"].pop("physics_info")
        data["inputs"]["selection_info"]=["n_jet","weight"]
        run.update(run_mode=["test"], checkpoint_dir=None, test_output=f"{stage}_scores.root")
    return out, data, model, run


def execute(out,data,model,run):
    from synapse.main import main
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for name,cfg in (("data",data),("model",model),("run",run)):
        path=out/(name+"_config.yaml")
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        files.append(str(path))
    uproot.open.defaults["handler"]=uproot.source.file.MemmapSource
    main(["-d",files[0],"-m",files[1],"-r",files[2]])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("train","validation","test"))
    ap.add_argument("--fold",type=int,required=True,choices=(0,1))
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path)
    args=ap.parse_args()
    root=args.run.resolve()
    cfg=configs(Path(__file__).resolve().parent,root/"dataset",root/"output"/f"fold_{args.fold}",args.fold,args.stage,args.checkpoint)
    execute(*cfg)
