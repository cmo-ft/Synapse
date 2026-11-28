"""Run one fold-1 learning-curve point with the frozen network and validation schedule."""
import argparse
from functools import cached_property, partial
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
from torch.utils.data import DataLoader

from selection import BASE, SELECTIONS, STUDY, TRIALS, load_selections

FROZEN = BASE / "frozen"
sys.path[:0] = [str(FROZEN), str(FROZEN / "recipe")]
import launch
import synapse.main as driver
from synapse.core.data_module import DataModule


class FixedBudgetBatches:
    """Read contiguous slices of a stream of complete shuffled passes. Epochs can span passes."""

    def __init__(self, rows, draws, batch_size, seed, epoch):
        self.rows, self.draws, self.batch_size = rows, draws, batch_size
        self.seed, self.epoch = seed, epoch
        self.cycle, self.order = None, None

    def __len__(self):
        return (self.draws + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        position = self.epoch() * self.draws
        end = position + self.draws
        while position < end:
            stop = min(end, position + self.batch_size)
            chunks = []
            while position < stop:
                cycle, offset = divmod(position, self.rows)
                if cycle != self.cycle:
                    rng = np.random.default_rng(np.random.SeedSequence([self.seed, cycle]))
                    self.order = rng.permutation(self.rows)
                    self.cycle = cycle
                take = min(self.rows - offset, stop - position)
                chunks.append(self.order[offset:offset + take])
                position += take
            yield chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class StatisticsDataModule(DataModule):
    def __init__(self, *args, record, mask_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.record = record
        self.mask_path = mask_path
        if self.train_file_list != record["files"] or self.val_file_list != [record["validation_file"]]:
            raise ValueError("File ordering or validation split differs from the audited selection")

    @cached_property
    def train_dataset(self):
        dataset = super().train_dataset
        if self.mask_path is not None:
            mask = np.load(self.mask_path, allow_pickle=False)
            if mask.dtype != bool or mask.shape != (len(dataset),) or mask.sum() != self.record["rows"]:
                raise ValueError("Training mask disagrees with the loaded files or selection manifest")
            for name in dataset.data:
                dataset.data[name] = dataset.data[name][mask]
            dataset.num_samples = self.record["rows"]
        if len(dataset) != self.record["rows"]:
            raise ValueError("Unexpected number of training rows")
        return dataset

    def train_dataloader(self):
        sampler = FixedBudgetBatches(len(self.train_dataset), self.record["draws_per_interval"],
                                     self.run_cfg.batch_size, self.run_cfg.seed,
                                     lambda: self.trainer.current_epoch)
        return DataLoader(self.train_dataset, batch_size=None, sampler=sampler,
                          num_workers=self.run_cfg.num_workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("point", choices=TRIALS)
    parser.add_argument("stage", choices=("train", "validation"))
    args = parser.parse_args()
    point, stage = args.point, args.stage
    record = load_selections()[point]
    workspace = STUDY / point
    checkpoint = None
    if stage == "validation":
        checkpoint, = (workspace / "fit/checkpoints").glob("BEST-*.ckpt")
    out, data, model, run = launch.configs(FROZEN / "recipe", BASE / "dataset",
                                           workspace, 1, stage, checkpoint)
    data["train_files"] = record["files"]
    data["weights"]["weight_balance_factors"] = record["class_balance"]
    if stage == "validation":
        if data["test_files"] != [record["validation_file"]]:
            raise ValueError("Scoring must use the common validation file")
        launch.execute(out, data, model, run)
        return
    data["test_files"] = []
    # An evicted worker resumes only from a completed validation checkpoint.
    candidates = list((out / "checkpoints").glob("model-epoch=*.ckpt"))
    if candidates:
        def epoch(path):
            return int(re.search(r"epoch=(?:epoch=)?(\d+)", path.name).group(1))
        latest = max(candidates, key=epoch)
        if epoch(latest) + 1 >= run["epochs"]:
            print("All training intervals are already complete.", flush=True)
            return
        run["resume_checkpoint"] = str(latest)
    mask = SELECTIONS / f"{point}.npy" if point in ("quarter", "half") else None
    factory = partial(StatisticsDataModule, record=record, mask_path=mask)
    print(f"{point}: {record['rows']:,} training rows, {record['draws_per_interval']:,} draws per validation", flush=True)
    with patch.object(driver, "DataModule", factory):
        launch.execute(out, data, model, run)


if __name__ == "__main__":
    main()
