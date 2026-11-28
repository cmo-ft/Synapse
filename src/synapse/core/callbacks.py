import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import uproot

import awkward as ak
import lightning as L
import numpy as np
import torch

from .config import DataConfig, ModelConfig, RunConfig
from .fileio import write_file

_logger = logging.getLogger("SynapseLogger")

class SaveTestOutputs(L.Callback):
    """
    Callback to save test outputs after the test ends.
    """
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, run_cfg: RunConfig, output_filepath: str):
        super().__init__()
        self.test_outputs = []
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.output_filepath = output_filepath
        self.stream_scores = getattr(run_cfg, "stream_scores", False)
        self.stream_file = self.stream_tree = None
        if not self.output_filepath.endswith(".root"):
            raise ValueError("Output filepath doesn't end with .root extension.")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect outputs from the test step.
        """
        labels = {k: v.numpy(force=True) for k,v in batch[1].items()}
        self.test_outputs.append(
            {
                "scores": outputs["scores"],
                "labels": labels,
                "weight": outputs["weight"],
                "spectators": outputs.get("spectators", {}),
            }
        )

        if self.stream_scores and len(self.test_outputs) >= 16:
            self._flush()

    def on_test_end(self, trainer, pl_module):
        if self.test_outputs:
            self._flush()
        if self.stream_file is not None:
            self.stream_file.close()
            Path(self.output_filepath + ".partial").replace(self.output_filepath)
            self.stream_file = self.stream_tree = None

    def on_exception(self, trainer, pl_module, exception):
        if self.stream_file is not None:
            self.stream_file.close()
            self.stream_file = self.stream_tree = None

    def _flush(self):
        if not self.test_outputs:
            _logger.warning("No test outputs collected.")
            return

        all_scores = np.concatenate([o["scores"] for o in self.test_outputs], axis=0)
        all_labels = {k: np.concatenate([o["labels"][k] for o in self.test_outputs], axis=0) for k in self.test_outputs[0]["labels"].keys()}
        all_weights = np.concatenate([o["weight"] for o in self.test_outputs], axis=0)
        all_spectators = {k: np.concatenate([o["spectators"][k] for o in self.test_outputs], axis=0) for k in self.test_outputs[0]["spectators"].keys()}

        _logger.info(f"Saving test outputs: {len(self.test_outputs)} batches collected.")
        self.save_to_root(all_scores, all_labels, all_weights, all_spectators)
        self.test_outputs.clear()

    def save_to_root(self, scores, labels, weights, spectators):
        """
        Save the test outputs to a ROOT file.
        """
        output = {}
        output_names = self.model_cfg.get('output_names', [])
        for idx, name in enumerate(output_names):
            output[name] = scores[:, idx]
        for idx in range(len(output_names), scores.shape[1]):
            output[f"extra_output_{idx}"] = scores[:, idx]

        # for label_name in labels.keys():
        #     output[label_name] = labels[label_name]
        if "_class_label" in labels.keys():
            categorical = self.data_cfg.labels["categorical"]
            if isinstance(categorical, str):
                output[categorical] = labels["_class_label"]
            else:
                for idx, label_name in enumerate(categorical):
                    output[label_name] = labels["_class_label"] == idx
        #         output[label_name + "_score"] = scores[:, idx]
        #     labels.pop("_class_label")

        # if len(labels.keys()) > 0:
        #     if self.model_cfg.model_params.get("num_classes"):
        #         n_classes = self.model_cfg.model_params["num_classes"]
        #     else:
        #         n_classes = len(self.data_cfg.labels.get("categorical",[]))
        #     for idx, label_name in enumerate(self.data_cfg.labels["continuous"]):
        #         output[label_name] = labels[label_name]
        #         output[label_name + "_pred"] = scores[:, n_classes + idx]

        if self.data_cfg.get('weights', {}).get('balance_weights', False):
            output["_balanced_weights"] = weights
        else:
            output["_weights"] = weights
        output.update(spectators)
        if self.stream_scores:
            if self.stream_file is None:
                self.stream_file = uproot.recreate(self.output_filepath + ".partial",compression=uproot.LZ4(4))
                schema = {k: np.dtype((v.dtype, v.shape[1:])) for k,v in output.items()}
                self.stream_tree = self.stream_file.mktree("tree", schema)
            self.stream_tree.extend(output)
            return
        write_file(self.output_filepath, ak.Array(output))

        _logger.info(f"Test outputs saved to {self.output_filepath}.")

class PhysicsEpochLog(L.Callback):
    """Publish one atomic record for each completed validation epoch."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.started = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 512 == 0:
            progress = dict(epoch=trainer.current_epoch+1, completed_batches=batch_idx+1,
                            batches=trainer.num_training_batches, steps=trainer.global_step,
                            loss=float(outputs["loss"].detach()), time=datetime.now(timezone.utc).isoformat())
            temp = self.directory / "progress.tmp"
            temp.write_text(json.dumps(progress)+"\n")
            temp.replace(self.directory / "progress.json")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        values = {k: float(v) for k, v in trainer.callback_metrics.items()
                  if k.startswith("val_") and k.endswith("_epoch") and v.numel() == 1}
        epoch = trainer.current_epoch + 1
        record = dict(epoch=epoch, steps=trainer.global_step, seconds=time.monotonic()-self.started,
                      completed=datetime.now(timezone.utc).isoformat(), **values)
        out = self.directory / f"epoch_{epoch:03d}.json"
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2)+"\n")
        tmp.replace(out)
        _logger.info("Completed training epoch %d: %s", epoch, values)


class SaveONNX(L.Callback):
    """
    Callback to save test outputs after the test ends.
    """
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, run_cfg: RunConfig, onnx_path: str):
        super().__init__()
        self.test_outputs = []
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.onnx_path = onnx_path
        if not self.onnx_path.endswith(".onnx"):
            raise ValueError("ONNX filepath doesn't end with .onnx extension.")

    def on_test_end(self, trainer, pl_module):
        """
        Save the model to ONNX format after the test ends.
        """
        _logger.info("Saving model to ONNX format...")
        pl_module.eval()
        model = pl_module.to('cpu')

        dummy_input = []

        for feat_name, feat_list in self.data_cfg.inputs.items():
            # temporary hack
            if "object_label" == feat_name or "aux_task" == feat_name or 'extra_event_info' == feat_name or 'extra_obj_info' == feat_name:
                continue
            feat_shape = None
            if feat_name == "evt_feats":
                feat_shape = (1, len(feat_list)) # e.g. (1, 10) for 10 event features
            else:
                feat_shape = (1, len(feat_list), 6) # e.g. (1, x, 6) for x features per particle. Hardcoded 6 for now.
            dummy_input.append(torch.randn(feat_shape, dtype=torch.float32))
        dummy_input = tuple(dummy_input)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            self.onnx_path,
            export_params=True,
        )
        _logger.info(f"Model saved to {self.onnx_path}")
