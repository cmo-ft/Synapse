"""Exercise a tiny real fold-1 fit, checkpoint restore, scoring and auxiliary evaluation on CPU."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import uproot
from paths import RUN


def subset(src,dst):
    with uproot.open(src,handler=uproot.source.file.MemmapSource) as root, uproot.recreate(dst) as out:
        tree=root["tree"]
        a=tree.arrays(["class_id","partOrigin","relationZ","wAssignment","isWZ"],library="np")
        masks=[a["class_id"]==k for k in (0,1,2)]
        masks += [(a["partOrigin"][:,:3]==k).any(1) for k in (1,2,3,4)]
        masks += [a["wAssignment"].astype(bool),a["isWZ"].astype(bool)&((a["relationZ"]>0).sum(1)>=2)]
        rows=np.unique(np.concatenate([np.flatnonzero(mask)[:48] for mask in masks]))
        writer=None
        for window in np.unique(rows//1024):
            start=int(window)*1024
            selected=rows[rows//1024==window]-start
            data=tree.arrays(entry_start=start,entry_stop=min(start+1024,tree.num_entries),library="np")
            batch={key:v[selected] for key,v in data.items()}
            if writer is None: writer=out.mktree("tree",{k:np.dtype((v.dtype,v.shape[1:])) for k,v in batch.items()})
            writer.extend(batch)
        return len(rows)


def main():
    frozen=RUN/"frozen"
    sys.path.insert(0,str(frozen))
    sys.path.insert(0,str(frozen/"recipe"))
    from launch import configs,execute
    from evaluate_aux import main as evaluate
    names=("3fold_0.root","wzcr_0.root","3fold_2.root","3fold_1.root")
    report={}
    with TemporaryDirectory(prefix="synapse-higgs-smoke-") as folder:
        tmp=Path(folder)
        report["mini_sample_rows"]={name:subset(RUN/"dataset"/name,tmp/name) for name in names}
        checkpoint=None
        for stage in ("train","validation","test"):
            out,data,model,run=configs(frozen/"recipe",RUN/"dataset",tmp/"outputs",1,stage,checkpoint)
            for key in ("train_files","val_files","test_files"):
                data[key]=[str(tmp/Path(name).name) for name in data[key]]
            run.update(device="cpu",epochs=1,batch_size=128,n_devices=1,num_workers=0)
            execute(out,data,model,run)
            if stage=="train":
                checkpoint,=(out/"checkpoints").glob("BEST-*.ckpt")
                report["checkpoint"]=checkpoint.name
                continue
            scores=out/f"{stage}_scores.root"
            with uproot.open(scores) as pred,uproot.open(data["test_files"][0]) as source:
                cols=["eventNumber","dsid","ancestryHiggs","relationHiggs","higgsPairKnown","higgsPairSame"]
                y=pred["tree"].arrays(cols+["p_signal","p_pair_higgs","valid","isMET"],library="np")
                original=source["tree"].arrays(cols,library="np")
                for key in cols: np.testing.assert_array_equal(y[key],original[key])
                assert np.isfinite(y["p_signal"]).all()
                i,j=np.triu_indices(y["valid"].shape[1],1)
                eligible=y["valid"].astype(bool)&~y["isMET"].astype(bool)
                allow=eligible[:,i]&eligible[:,j]
                assert np.isfinite(y["p_pair_higgs"][allow]).all()
            dest=tmp/(stage+"_aux.json")
            evaluate(scores,dest)
            report[stage]=json.loads(dest.read_bytes())["tasks"]["pair_higgs"]
    out=RUN/"smoke_check.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
