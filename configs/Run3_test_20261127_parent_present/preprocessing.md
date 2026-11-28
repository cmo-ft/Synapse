# Rebuild the Run 3 parent_present training folds

The six input files for training are produced from the 3l0tau nominal MC ntuples. The selected CAF trees supply the reconstructed events. The readers use the matching CAF job plan to find the original EasyJet files for auxiliary object and parent labels. All commands below run from the Synapse repository root.

## Inputs

| Input | Path relative to Synapse |
|---|---|
| Selected `sr`, `extconvcr`, and `wzcr` ROOT parts | `../../caf/hhml3l0tau/output/nominal-run3-20261122/ntuples/run3/parts/NOSYS/` |
| Frozen plan mapping MC jobs and samples to EasyJet files | `../../caf/hhml3l0tau/work/nominal-run3-20261122/plan/plan.json` |
| EasyJet `AnalysisMiniTree` files | Paths listed in the plan's `jobs[].files` |

The plan and both sets of ntuples are external production data. The Git branch contains the reconstruction and truth-matching code, including `configs/run3_common/physics.hxx` copied from this CAF production's `source/include/physics.hxx` (SHA-256 `efcb8ff18810405498f82b03749f778cec304d3c3df1df7826c1458837609586`). The recipe also requires Python, PyROOT, uproot, awkward, NumPy and PyYAML. At CERN the production workers used this environment.

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## Build order

Use a fresh checkout with access to the production inputs for a full rebuild. Each stage writes under `run_dir/` in the directory named after its recipe. Keep the intermediate datasets and shards through the last step. Stages 2–5 use job groups. The commands below assign the entire stage to one worker. For batch processing, divide the same stage over group indices 0 through N−1 with the same `--groups N`, and finish all groups before its merge.

1. **Base MC folds.** The first pass reads the selected SR and ExtConvCR events and writes three files with the original 35 event features, per-lepton truth flags, nominal weights and event identifiers. The remainder is `eventNumber % 3`. The additional forward-jet count is introduced in the next stage.

   ```bash
   python3 -B configs/Run3_test_20261123/prepare_folds.py
   ```

2. **Physics features and WZCR.** Each MC job joins the selected events to EasyJet and writes one physics shard. The WZCR sample uses a fixed hash of the DSID and event number to retain approximately one eighth of events. The merge checks the SR + ExtConvCR rows against the initial folds and publishes the three additional WZCR files.

   ```bash
   python3 -B configs/Run3_test_20261124_physics/prepare_physics.py --group 0 --groups 1
   python3 -B configs/Run3_test_20261124_physics/merge_physics.py
   ```

3. **Parent relations.** The raw reader writes parent IDs and candidate assignments by job. The WZCR audit independently counts additional pairs from the strict mass fallback. The merge checks those counts, joins the labels in the original event order and drops the lepton impact-parameter branches.

   ```bash
   python3 -B configs/Run3_test_20261125_relations/prepare_relations.py --group 0 --groups 1
   python3 -B configs/Run3_test_20261125_relations/audit_wzcr_mass_gain.py
   python3 -B configs/Run3_test_20261125_relations/merge_relations.py
   ```

4. **Object ancestry.** Read the indexed MC parent masks and daughter evidence, then publish the object ancestry and W-assignment labels.

   ```bash
   python3 -B configs/Run3_test_20261126_ancestry/prepare_raw.py --group 0 --groups 1
   python3 -B configs/Run3_test_20261126_ancestry/prepare_folds.py
   ```

5. **Higgs decay tree.** Match reconstructed objects to recorded Higgs children and grandchildren. Publish the object labels, same-Higgs pair bits and known-pair bits together with their audit.

   ```bash
   python3 -B configs/Run3_test_20261127_higgs_tree/prepare_raw.py --group 0 --groups 1
   python3 -B configs/Run3_test_20261127_higgs_tree/publish.py
   ```

6. **Lepton fallback and Higgs presence.** The first publisher retrieves the required EasyJet lepton mask evidence through `mask_sources.py`. The second applies the Higgs-presence condition and checks the six outputs. The detailed matching rules and remaining ambiguities are in `configs/Run3_test_20261127_higgs_lepton_fallback/truth_labels.md`.

   ```bash
   python3 -B configs/Run3_test_20261127_higgs_lepton_fallback/publish.py
   python3 -B configs/Run3_test_20261127_parent_present/prepare_dataset.py
   ```

## Outputs

The training dataset is in `run_dir/Run3_test_20261127_parent_present/dataset/`. For each remainder 0, 1, and 2, `3fold_<remainder>.root` contains SR + ExtConvCR and `wzcr_<remainder>.root` contains control events. The same directory contains `summary.json` and `audit.json`. In the reference fold 1 training, remainder 0 provides training events, 2 provides validation and 1 provides test. WZCR rows supervise the auxiliary tasks during training.
