# Next Synapse iteration

## Per-object ancestry

Study four independent sigmoid outputs per reconstructed lepton or jet, one each for Higgs, top, W and Z ancestry. Use the same transformer tokens as the shared-parent matrices and train both tasks together. A lepton from H → W → ℓ can be positive for both Higgs and W.

For `relationHiggs = [1, 1, 2]`, the proposed lepton ancestry targets are `[1, 1, 1]`. The existing pair targets for `(ℓ1, ℓ2)`, `(ℓ1, ℓ3)` and `(ℓ2, ℓ3)` are `[1, 0, 0]`. This gives ℓ3 positive ancestry supervision even if it has no partner. Prefer a separate per-object head to make the ancestry scores easy to inspect. Using the matrix diagonal is an alternative that requires the same truth labels.

### Open label question

Define a positive as a verified descendant, a negative as a verified nonmember, and mask unresolved ancestry. Across the three v06 folds, the valid lepton and jet slots contain zero explicit nonmember IDs for Higgs, top and W, and 746,631 for Z.

- Audit the EasyJet ancestry masks and truth matches for each parent. Check when a zero mask establishes nonmembership and when the match or branch is missing. Also check whether a multiple-parent match can establish ancestry even when the parent instance is ambiguous.
- Look for verified non-Higgs leptons and jets **within single-H and HH events**. Count positives, confirmed negatives and unknowns by parent, object type and process before choosing the training labels.

## Test class balancing for W pairs

On the fold-0 training set, positive pairs receive 36.6% of the Higgs pair weight, 33.3% for top, 10.0% for W and 33.4% for Z. These fractions include the training event weights and the average over known pairs within each event.

Compare the current recipe with a run that gives positive and negative W pairs 50% of the W loss weight each. Fit the factors once on the training fold (about 5.0 for positives and 0.556 for negatives). Use the same seed and fold for both runs and keep the W loss coefficient at `0.02`.

Judge the change by fold-0 validation HH significance with `p_signal` as the fit observable in the `n_jet >= 1` and `p_extconv_bkg < 0.8` signal region.
