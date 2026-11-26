from functools import partial
import inspect
from typing import Dict, Any
import logging

import lightning as L
import torch
from torch.optim import Adam, AdamW, RAdam

from .config import RunConfig
from .tools import dynamic_import, is_scalar
from .optimizers import Ranger

_logger = logging.getLogger("SynapseLogger")

class ModelModule(L.LightningModule):
    def __init__(
            self,
            run_cfg: RunConfig,
            model_class: str,
            model_params: Dict[str, Any],
            loss_fn: str, # TODO: merge loss_fn and loss_params into a single dict
            loss_params: Dict[str, Any] | None = None,
            training_step_func: str = "default",
            validation_step_func: str = "default",
            test_step_func: str = "default",
            optimizer: str = "adam",
            start_lr: float = 1e-3,
            lr_scheduler: str | None = None,
            metrics: Dict[str, dict] | None = None,
            mean_train_weight: float | None = None,
            binary_object_output_names: list[str] | None = None,
            relation_output_names: list[str] | None = None,
    ) :
        """
        Initialize LightningModule wrapper.

        Args:
            run_cfg (RunConfig): Configuration for the run, including batch size, epochs, etc.
            model_class (str): The fully qualified class name of the model to be used.
            model_params (dict): Model parameters to be passed to the model.
            loss_fn (str): The fully qualified name of the loss function to be used.
            loss_params (dict): Parameters for the loss function.
            training_step_func (str): Custom training step function. Default is "default".
            validation_step_func (str): Custom validation step function. Default is "default".
            test_step_func (str): Custom test step function. Default is "default".
            optimizer (str): Optimizer to be used.
            start_lr (float): Initial learning rate for the optimizer.
            lr_scheduler (str): Learning rate scheduler to be used.
            metrics (dict): Dictionary of metric names and their corresponding functions.
            mean_train_weight (float): Training-fold loss normalization restored from a checkpoint.
            binary_object_output_names (list): Ordered ROOT branch names for the per-object probabilities.
            relation_output_names (list): Ordered ROOT branch names for the upper-triangle pair probabilities.
        """
        super().__init__()
        self.run_cfg = run_cfg
        if binary_object_output_names is None:
            # Older saved recipes kept these names in the run configuration.
            binary_object_output_names = run_cfg.get("binary_object_output_names", [])
        self.binary_object_output_names = binary_object_output_names
        self.relation_output_names = relation_output_names or []

        model_cls = dynamic_import(model_class)

        if not issubclass(model_cls, torch.nn.Module):
            raise TypeError(f"model_class: {model_class} is not the subclass of torch.nn.Module")

        self.model = model_cls(**model_params)

        loss_params = dict(loss_params or {})
        self._use_fold_mean = loss_params.get("mean_train_weight") == "auto"
        if self._use_fold_mean:
            del loss_params["mean_train_weight"]
            self.save_hyperparameters("mean_train_weight")
        loss_function = dynamic_import(loss_fn)
        if inspect.isclass(loss_function) and issubclass(loss_function, torch.nn.Module):
            if loss_params:
                self.loss_fn = loss_function(**loss_params)
            else:
                self.loss_fn = loss_function()
        elif inspect.isfunction(loss_function):
            if loss_params:
                self.loss_fn = partial(loss_function, **loss_params)
            else:
                self.loss_fn = loss_function
        else:
            raise TypeError(f"loss_function: {loss_fn} is not a valid loss function or class")

        self._training_step = self._default_training_step
        if training_step_func != "default":
            training_step_fn = dynamic_import(training_step_func)
            if inspect.isfunction(training_step_fn):
                self._training_step = training_step_fn
            else:
                raise TypeError(f"Invalid training step function: {training_step_func}")

        self._validation_step = self._default_validation_step
        if validation_step_func != "default":
            validation_step_fn = dynamic_import(validation_step_func)
            if inspect.isfunction(validation_step_fn):
                self._validation_step = validation_step_fn
            else:
                raise TypeError(f"Invalid validation step function: {validation_step_func}")

        self._test_step = self._default_test_step
        if test_step_func != "default":
            test_step_fn = dynamic_import(test_step_func)
            if inspect.isfunction(test_step_fn):
                self._test_step = test_step_fn
            else:
                raise TypeError(f"Invalid test step function: {test_step_func}")

        self.optimizer = optimizer.lower()
        self.start_lr = start_lr
        self.lr_scheduler = lr_scheduler.lower() if lr_scheduler else None

        self.metrics = {}
        if metrics:
            for metric_name, metric_fn_dict in metrics.items():
                metric_fn = dynamic_import(metric_fn_dict['function'])
                if inspect.isclass(metric_fn) and issubclass(metric_fn, torch.nn.Module):
                    if metric_fn_dict.get('params'):
                        self.metrics[metric_name] = {"fn":metric_fn(**metric_fn_dict['params'])}
                    else:
                        self.metrics[metric_name] = {"fn":metric_fn()}
                elif inspect.isfunction(metric_fn):
                    if metric_fn_dict.get('params'):
                        self.metrics[metric_name] = {"fn":partial(metric_fn, **metric_fn_dict['params'])}
                    else:
                        self.metrics[metric_name] = {"fn":partial(metric_fn)}
                else:
                    raise TypeError(f"Invalid metric function: {metric_fn}")
                # TODO: move this to model_config validation
                self.metrics[metric_name]['on_step'] = metric_fn_dict.get('on_step', False)
                self.metrics[metric_name]['on_epoch'] = metric_fn_dict.get('on_epoch', True)
                self.metrics[metric_name]['stages'] = metric_fn_dict.get('stages', ["train", "val", "test"])
                # TODO: validate that metrics has 'is_monitor' must has 'mode' and its stages contains 'val'
                self.metrics[metric_name]['is_monitor'] = metric_fn_dict.get('is_monitor', False)
                self.metrics[metric_name]['mode'] = metric_fn_dict.get('mode')

        # TODO: move this validation to model_config validation
        # validate is_monitor flag is unique
        is_monitor_count = sum(1 for m in self.metrics.values() if m['is_monitor'])
        if is_monitor_count > 1:
            raise ValueError("Only one metric can be marked as monitor (is_monitor=True).")

        self._cache_epoch_outputs = {
            stage: any(
                stage in metric_cfg["stages"] and metric_cfg["on_epoch"]
                for metric_cfg in self.metrics.values()
            )
            for stage in ("train", "val", "test")
        }
        self.step_outputs = {
            "train": [],
            "val": [],
            "test": []
        }

    def setup(self, stage: str) -> None:
        if stage == "fit" and self._use_fold_mean:
            self.hparams.mean_train_weight = self.trainer.datamodule.train_dataset.mean_weight
            _logger.info("Mean balanced training event weight: %.10g", self.hparams.mean_train_weight)

    def on_train_start(self) -> None:
        # Fit after checkpoint restoration so a resumed run uses this fold's object weights.
        if hasattr(self.loss_fn, "prepare_training_fold"):
            dataset = self.trainer.datamodule.train_dataset
            data = dataset.data
            self.loss_fn.prepare_training_fold(data, data["mask"], data[dataset.data_cfg.weight_key])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Universal forward method to call the model.
        """
        return self.model(*args, **kwargs)

    def _shared_default_step(self, batch: Any, stage: str):
        x, y, w, s = batch
        object_label = x.get('object_label', None)
        if object_label is not None:
            x.pop('object_label')
        object_mask = x['mask']
        # aux_task = x.get('aux_task', None)
        # if aux_task is not None:
        #     x.pop('aux_task')
        extra_event_info = x.get('extra_event_info', None)
        if extra_event_info is not None:
            x.pop('extra_event_info')
        extra_obj_info = x.get('extra_obj_info', None)
        if extra_obj_info is not None:
            x.pop('extra_obj_info')


        inputs = [x[k] for k in x.keys()]
        labels = [value for name, value in y.items() if name != 'objects']
        weight = [w[k] for k in w.keys()][0]

        logits = self(*inputs)
        loss_kwargs = dict(weight=weight, extra_event_info=extra_event_info,
                           extra_obj_info=(extra_obj_info, object_label, object_mask))
        if self._use_fold_mean:
            loss_kwargs["mean_train_weight"] = self.hparams.mean_train_weight
        if 'objects' in y:
            loss_kwargs.update(object_targets=y['objects'], object_mask=object_mask)
        loss, scores = self.loss_fn(logits, *labels, **loss_kwargs)

        logits, logits_aux, obj_logits = logits

        # loss, scores = self.loss_fn(logits, *labels, weight = weight, extra_event_info=extra_event_info, extra_obj_info=(extra_obj_info, object_mask))


        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=weight.shape[0])
        for metric_name, metric_fn_dict in self.metrics.items():
            if stage in metric_fn_dict["stages"] and metric_fn_dict["on_step"]:
                metric = metric_fn_dict['fn'](logits, *labels, weight = weight)
                if is_scalar(metric):
                    self.log(f"{stage}_{metric_name}_step", metric, on_step=True, prog_bar=True, logger=True)

        if self._cache_epoch_outputs[stage]:
            self.step_outputs[stage].append(
                {
                    "logits": logits.detach().cpu(),
                    "labels": [label.detach().cpu() for label in labels],
                    "weight": weight.detach().cpu(),
                }
            )

        if stage == "test":
            columns = {k: value.numpy(force=True) for k, value in s.items()}
            if self.relation_output_names:
                matrices = logits_aux["shared_parent_logits"]
                # Store (0,1), (0,2), ..., (1,2), ... once per symmetric matrix.
                max_objects = matrices.size(-1)
                i, j = torch.triu_indices(max_objects, max_objects, offset=1, device=matrices.device)
                eligible = object_mask[:, 0].bool() & ~s["isMET"].bool()
                valid_pairs = eligible[:, i] & eligible[:, j]
                probabilities = matrices[:, :, i, j].sigmoid().masked_fill(~valid_pairs[:, None, :], -1.)
                for name, values in zip(self.relation_output_names, probabilities.unbind(1), strict=True):
                    columns[name] = values.numpy(force=True)
            names = self.binary_object_output_names
            if names:
                batch_size, _, max_objects = object_mask.shape
                if obj_logits.shape != (batch_size * max_objects, len(names)):
                    raise ValueError("Object logits must match the configured names and padded input dimensions")
                probs = obj_logits.reshape(batch_size, max_objects, len(names)).sigmoid().numpy(force=True)
                columns.update({name: probs[:, :, i] for i, name in enumerate(names)})
            return {
                "loss": loss,
                "scores": scores.numpy(force=True),
                "weight": weight.numpy(force=True),
                "spectators": columns
            }

        return loss

    def _default_training_step(self, batch):
        return self._shared_default_step(batch, "train")

    def _default_validation_step(self, batch):
        return self._shared_default_step(batch, "val")

    def _default_test_step(self, batch):
        return self._shared_default_step(batch, "test")

    def training_step(self, batch, batch_idx):
        return self._training_step(batch)

    def validation_step(self, batch, batch_idx):
        return self._validation_step(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch)

    def on_train_epoch_start(self) -> None:
        _logger.info(f"======= Training epoch {self.current_epoch} =======")
        restart = self.run_cfg.get("cosine_restart")
        if restart and self.current_epoch == restart["epoch"]:
            scheduler = self.lr_schedulers()
            if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                raise ValueError("cosine_restart requires a cosine scheduler")
            optimizer = scheduler.optimizer
            peak = float(restart["lr"])
            if peak <= 0:
                raise ValueError("The cosine restart learning rate must be positive")
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"] = peak
            # Lightning retains the scheduler object restored with the Adam state.
            cycle = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs - self.current_epoch, eta_min=scheduler.eta_min
            )
            scheduler.load_state_dict(cycle.state_dict())
        if restart:
            _logger.info("Optimizer step %d, learning rate %.9g", self.global_step,
                         self.trainer.optimizers[0].param_groups[0]["lr"])

    def on_validation_epoch_start(self) -> None:
        _logger.info(f"======= Validating epoch {self.current_epoch} =======")

    def _shared_on_epoch_end(self, stage: str):
        if not self._cache_epoch_outputs[stage]:
            return

        outputs = self.step_outputs[stage]
        if not outputs:
            return

        all_logits = torch.cat([o["logits"] for o in outputs])
        all_labels = []
        for i in range(len(outputs[0]["labels"])):
            all_labels.append(torch.cat([o["labels"][i] for o in outputs]))
        all_weights = torch.cat([o["weight"] for o in outputs])

        for metric_name, metric_fn_dict in self.metrics.items():
            if stage in metric_fn_dict["stages"] and metric_fn_dict["on_epoch"]:
                metric = metric_fn_dict['fn'](all_logits, *all_labels, weight=all_weights)
                _logger.info(f"{stage}_{metric_name}_epoch: \n{metric}")
                if is_scalar(metric):
                    self.log(f"{stage}_{metric_name}_epoch", metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.step_outputs[stage].clear()

    def on_train_epoch_end(self):
        self._shared_on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._shared_on_epoch_end("val")

    def on_test_epoch_end(self):
        self._shared_on_epoch_end("test")

    def configure_optimizers(self):
        # TODO: record LR
        optimizer_type = self.optimizer
        lr_scheduler_type = self.lr_scheduler
        lr = self.start_lr
        # TODO: make optimizer and scheduler parameters configurable, following the same way as loss function loading
        if optimizer_type == "adam":
            optimizer = Adam(self.parameters(), lr=lr)
        elif optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=lr)
        elif optimizer_type == "radam":
            optimizer = RAdam(self.parameters(), lr=lr)
        elif optimizer_type == "ranger":
            optimizer = Ranger(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        scheduler_dict = {}
        if lr_scheduler_type:
            if lr_scheduler_type == "steps":
                lr_step = round(self.run_cfg.epochs / 3)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[lr_step, 2 * lr_step], gamma=0.1,
                    last_epoch=-1) # FIXME: last_epoch=-1 means it will start from the beginning, fix after implementing checkpoint loading
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            elif lr_scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.run_cfg.epochs, eta_min=0.0, last_epoch=-1)
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            elif lr_scheduler_type == "one-cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=lr, total_steps=self.run_cfg.epochs * 2600, # length of dataset / batch size. Hardcoded for now
                    pct_start=0.1, anneal_strategy='cos', cycle_momentum=False, last_epoch=-1)
                scheduler_dict = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }


        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict
        }
