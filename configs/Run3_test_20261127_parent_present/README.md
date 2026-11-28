# Run 3 parent_present ParT

The reference is the completed **fold 1**, trained on simulated 3l0tau events. The checkpoint at epoch 43 of 50 maximizes validation significance in three disjoint HH-efficiency bins. See `model_config.yaml`, `data_config.yaml`, and `run_config.yaml` in this directory for the exact training definitions.

## Changes since 0073327

| Change | Implementation and interpretation |
|---|---|
| Model and control region | The 36 event features and 14 object features feed a five-block ParT. Four shared-parent matrices use block 4 and contribute to block 5 attention. New per-object Higgs, top, W and Z ancestry outputs and a WZ recoil classifier add physics supervision. WZCR rows contribute to the Z-pair and recoil tasks. The main HH/background/conversion loss uses SR and ExtConvCR. The event score uses reconstructed inputs. EveNet and XGBoost are evaluation references. |
| Higgs truth | The decay-tree recipe matches recorded children and grandchildren. The lepton fallback fills supported missing labels using EasyJet evidence. In `parent_present`, Higgs-pair supervision requires a recorded Higgs and known pair evidence. A known negative inside a Higgs event can contribute even if there is no positive reconstructed pair. Events with absent or unresolved Higgs records have their known-pair bits cleared. Object ancestry labels, positive pair bits, and W/Z/top labels follow the preceding dataset. |
| Training | The input statistics are accumulated in float64 on the training sample. Standardized features are clipped at ±12. Batch-indexed loading and streamed ROOT output lower the memory needed by the training and application workers. Fold 1 uses AdamW, an initial learning rate of 0.0008, cosine decay, batch size 768, and 50 epochs. |
| Checkpoint and evaluation | After each epoch the worker records validation loss, AUC, background rejection at 20% HH efficiency, and three-bin significance. The latter combines the 0–20%, 20–30%, and 30–50% bands in quadrature using signed yields and background MC variance Σw². The selected checkpoint is applied separately to validation and test. The native hhml_eval Trafo60 calculation is a separate benchmark. |

The `prepare_dataset.py` audit verifies all six published ROOT files against their inputs. In the fold 1 SR + ExtConvCR training file, the Higgs-pair task applies to 643,081 events after the presence requirement. It retains 1,661,748 positive pairs. The training and validation implementations are in `src/synapse/`; every model source listed in the completed run's frozen manifest was checked against this branch during packaging.

## Recipe lineage

The paths below are relative to `configs/`. For the commands to rebuild all six labelled folds from the selected CAF ntuples and the indexed EasyJet files, see `preprocessing.md` in this directory.

| Directory | Contents |
|---|---|
| `Run3_test_20261123/` | Initial SR and ExtConvCR MC folds, object truth and original 35-event-feature schema. |
| `Run3_test_20261124_physics/` | Reconstructed jet inputs, control-region sampling, EasyJet matching and six-file merge. |
| `Run3_test_20261125_relations/` | Parent identities, the WZ control-region assignment audit and publication of the impact-parameter-free folds. |
| `Run3_test_20261126_ancestry/` | Ancestry targets, WZ supervision, training launcher, evaluation and control-region checks. |
| `Run3_test_20261127_higgs_tree/` | Explicit Higgs daughters and grandchildren, ambiguous matches, and packed pair labels. |
| `Run3_test_20261127_higgs_lepton_fallback/` | EasyJet lepton fallback, consistency checks, documented truth-label limitations, migration diagnostics and comparisons. Start with `truth_labels.md`. |
| `Run3_test_20261127_parent_present/` | Presence requirement on Higgs-pair supervision, reference checkpoint selection and validation/test comparisons. |
| `run3_common/` | Shared reconstruction headers, process groups, rejection and Trafo60 plotting, and the native hhml_eval entry point. |
| `Run3_test_20261127_triboson/` | Separate nine-input study using additional reconstructed HH-candidate observables. |
| `Run3_test_20261128_training_statistics/` | Nested event subsets and the doubled-training-set study, evaluated on a common validation split. |

## Fold 1 sample

| Role | eventNumber % 3 | SR + ExtConvCR rows | WZCR rows used |
|---|---:|---:|---:|
| Training | 0 | 2,296,499 | 518,851 |
| Validation | 2 | 2,297,211 | 0 |
| Test | 1 | 2,295,545 | 0 |

## Train or apply from a checkout

Activate the Synapse Python environment from the repository root. For an already published dataset, stage the program into a **new** run directory and point it at the folder containing `3fold_{0,1,2}.root`, `wzcr_{0,1,2}.root` and `summary.json`.

```bash
run="$PWD/run_dir/parent_present_reproduction"
python configs/Run3_test_20261127_parent_present/stage_run.py --run "$run"
ln -s /path/to/published/parent_present/dataset "$run/dataset"
export PYTHONPATH="$run/frozen:${PYTHONPATH:-}"
python "$run/frozen/recipe/launch.py" train --fold 1 --run "$run"
```

To apply an existing best checkpoint to the published folds, use the same staged run and dataset. Set `checkpoint` to the full path of the checkpoint, for example the `model/best.ckpt` supplied in the review package.

```bash
checkpoint=/path/to/parent_present_fold1/model/best.ckpt
python "$run/frozen/recipe/launch.py" validation --fold 1 --run "$run" --checkpoint "$checkpoint"
python "$run/frozen/recipe/launch.py" test --fold 1 --run "$run" --checkpoint "$checkpoint"
```

`stage_run.py --reference /path/to/previous/run` selects and verifies sources from an archived predecessor. Without that option the staging code reads the branch and writes a SHA-256 manifest. These training commands require the labelled MC folds. The preprocessing recipes include the original CAF reconstruction header in `configs/run3_common/physics.hxx`. The external inputs are the selected MC ntuples, the matching CAF plan and the EasyJet files indexed by that plan.

## Read the transfer package

Start with `README.md` in the exported `parent_present_fold1/` directory. It includes the selected checkpoint, the exact frozen snapshot, both splits' scores used by hhml_eval, rejection curves against the previous NN/EveNet/XGBoost, background composition, Trafo60 results, and the branch history for offline inspection. The display names `validation` and `test` refer to the roles in the table above. The archived native evaluator internally calls the test split `held_out`.

With dependencies installed, `python -B examples/check_checkpoint.py .` in the exported directory checks the original event, object, recoil and pair scores on a small sample supplied from each split. For full evaluation in a fresh Synapse environment, install `pip install -e '.[review]'` from the checkout and make PyROOT available for native Trafo60.

To make the export from the completed run, use a new output directory.

```bash
python -B configs/Run3_test_20261127_parent_present/export_review.py --out review_exports/parent_present_fold1
tar -czf review_exports/parent_present_fold1.tar.gz -C review_exports parent_present_fold1
```
