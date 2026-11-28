"""Submit the indexed CPU readers or the fold-1 GPU training worker once."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from tempfile import mkdtemp
from paths import RECIPE, RUN


def submit(stage):
    RUN.mkdir(parents=True,exist_ok=True)
    marker=RUN/f"submission_{stage}.json"
    if marker.exists():
        print(marker.read_text(),flush=True)
        return
    spool=Path(mkdtemp(prefix=f"run3-ancestry-{stage}-",dir="/afs/cern.ch/user/c/cmo/private/workspace/hhml/submit-synapse"))
    if stage=="raw":
        args=f"{RECIPE}/raw_worker.sh --group $(group) --groups 16"
        resources='request_cpus = 1\nrequest_memory = 3500M\n+JobFlavour = "workday"'
        queue="queue group from (\n"+"\n".join(map(str,range(1,16)))+"\n)"
    else:
        args=f"{RUN}/worker.sh 1"
        resources='request_cpus = 8\nrequest_memory = 26000M\nrequest_gpus = 1\n+JobFlavour = "testmatch"'
        queue="queue 1"
    spec=spool/"job.sub"
    spec.write_text(f"""universe = vanilla
executable = /bin/bash
arguments = {args}
initialdir = {spool}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = False
transfer_output_files = ""
log = {spool}/events.log
output = {spool}/$(ClusterId).$(ProcId).out
error = {spool}/$(ClusterId).$(ProcId).err
MY.SendCredential = True
request_disk = 5000M
{resources}
{queue}
""")
    reply=subprocess.check_output(["condor_submit","-terse",str(spec)],cwd=spool,text=True).strip()
    if not re.fullmatch(r"\d+\.\d+(?:\s*-\s*\d+\.\d+)?",reply):
        raise RuntimeError(f"Inspect submission in {spool}: {reply}")
    data=dict(jobs=reply,submit_dir=str(spool),submitted=datetime.now(timezone.utc).isoformat())
    marker.write_text(json.dumps(data,indent=2)+"\n")
    print(marker.read_text(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("raw","train"))
    submit(p.parse_args().stage)
