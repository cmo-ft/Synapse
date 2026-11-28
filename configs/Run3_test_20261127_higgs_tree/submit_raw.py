"""Submit the indexed CPU readers for the Higgs decay labels."""
import json
import re
import subprocess
from pathlib import Path
from tempfile import mkdtemp
from paths import RECIPE, RUN

GROUPS = 24


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    receipt = RUN/"submission_raw.json"
    if receipt.is_file():
        print(receipt.read_text())
        return
    spool = Path(mkdtemp(prefix="higgs-tree-raw-",dir="/afs/cern.ch/user/c/cmo/private/workspace/hhml/submit-synapse"))
    spec = spool/"job.sub"
    spec.write_text(f"""universe = vanilla
executable = /bin/bash
arguments = {RECIPE}/raw_worker.sh --groups {GROUPS} --group $(ProcId)
initialdir = {spool}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_executable = False
transfer_output_files = ""
log = {spool}/events.log
output = {spool}/$(ClusterId).$(ProcId).out
error = {spool}/$(ClusterId).$(ProcId).err
MY.SendCredential = True
request_cpus = 1
request_memory = 4000M
request_disk = 1500M
+JobFlavour = "workday"
max_retries = 2
queue {GROUPS}
""")
    ids = subprocess.check_output(["condor_submit","-terse","-batch-name",RUN.name+"_cpu",str(spec)],cwd=spool,text=True).strip()
    if not re.fullmatch(r"\d+\.\d+\s*-\s*\d+\.\d+",ids):
        raise RuntimeError(f"Inspect submission in {spool}: {ids}")
    receipt.write_text(json.dumps(dict(jobs=ids,spool=str(spool),groups=GROUPS),indent=2)+"\n")
    print(receipt.read_text(),flush=True)


if __name__=="__main__":
    main()
