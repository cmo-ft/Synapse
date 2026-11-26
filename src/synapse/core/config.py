import os
import yaml
from abc import ABC, abstractmethod
from types import NoneType
from typing import Dict, Any, Type, List

from ..models.shared_parent import DEFAULT_SHARED_PARENT_VETO
from .tools import flatten_nested_list

class ConfigBase(ABC):
    """
    Base class for configuration management with validation and YAML I/O
    """

    def __init__(self, cfg_file_path: str):
        self._data: Dict[str, Any] = {}
        self._cfg_file_path = cfg_file_path
        self.load(cfg_file_path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with optional default
        """
        return self._data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        """Safely access configuration values"""
        data = object.__getattribute__(self, '_data')

        try:
            return data[name]
        except KeyError:
            available_keys = list(data.keys())
            raise AttributeError(
                f"'{self.__class__.__name__}' has no attribute '{name}'. "
                f"Available keys: {available_keys}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set configuration values with validation
        """
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._validate_and_set(name, value)

    def __getstate__(self):
        """Enable pickling for multiprocessing"""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Enable unpickling for multiprocessing"""
        self.__dict__.update(state)

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration
        """
        return key in self._data

    def __copy__(self):
        """
        Create a shallow copy of the configuration
        """
        new_instance = self.__class__(self._cfg_file_path)
        new_instance._data = self._data.copy()
        return new_instance

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the configuration
        """
        from copy import deepcopy
        new_instance = self.__class__(self._cfg_file_path)
        new_instance._data = deepcopy(self._data, memo)
        return new_instance

    @abstractmethod
    def validate(self) -> List[str]:
        """
        Validate configuration values

        Returns:
            List of validation error messages (empty if valid)
        """
        pass

    def load(self, file_path: str) -> None:
        """
        Load configuration from YAML file
        """
        self._cfg_file_path = file_path

        try:
            with open(file_path, 'r') as f:
                self._data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise SyntaxError(f"YAML syntax error in {file_path}: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {file_path}")
        except Exception as e:
            raise IOError(f"Error loading config file: {str(e)}")

        # Validate after loading
        errors = self.validate()
        if errors:
            raise ValueError(
                f"Config validation failed in {self.__class__.__name__}:\n" +
                "\n".join(f" - {err}" for err in errors)
            )

    def save(self, save_path: str) -> None:
        """
        Save configuration to YAML file
        """
        if save_path == self._cfg_file_path:
            raise ValueError("Cannot save to the same file as loaded configuration.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self._data, f, sort_keys=False, default_flow_style=False)
        except Exception as e:
            raise IOError(f"Error saving config file: {str(e)}")

    def update(self, values: Dict[str, Any]) -> None:
        """
        Update multiple configuration values with validation
        """
        for key, value in values.items():
            self._validate_and_set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary
        """
        return self._data.copy()

    def _validate_and_set(self, key: str, value: Any) -> None:
        """
        Validate and set a value, maintaining type safety
        """
        # Get expected type from validation specs
        expected_types = self._get_expected_types()

        if key in expected_types:
            expected_type = expected_types[key]
            # Check if it's a union type (tuple of types)
            if isinstance(expected_type, tuple):
                if not any(isinstance(value, t) for t in expected_type):
                    actual_type = type(value).__name__
                    raise TypeError(
                        f"{key} should be one of {[t.__name__ for t in expected_type]}, "
                        f"got {actual_type} with value {value}"
                    )
            elif not isinstance(value, expected_type):
                actual_type = type(value).__name__
                raise TypeError(
                    f"{key} should be {expected_type.__name__}, "
                    f"got {actual_type} with value {value}"
                )

        self._data[key] = value
        errors = self.validate()
        if errors:
            self._data.pop(key, None)  # Revert invalid change
            raise RuntimeError(
                f"Config validation failed after setting '{key}':\n" +
                "\n".join(f" - {err}" for err in errors)
            )

    def _check_types(self):
        """
        Check if all keys in the configuration match their expected types
        """
        errors = []
        # Get expected type from validation specs
        expected_types = self._get_expected_types()

        for key, expected_type in expected_types.items():
            if key in self._data:
                value = self._data[key]
                if isinstance(expected_type, tuple):
                    if not any(isinstance(value, t) for t in expected_type):
                        actual_type = type(value).__name__
                        errors.append(
                            f"{key} should be one of {[t.__name__ for t in expected_type]}, "
                            f"got {actual_type} with value {value}"
                        )
                elif not isinstance(value, expected_type):
                    actual_type = type(value).__name__
                    errors.append(
                        f"{key} should be {expected_type.__name__}, "
                        f"got {actual_type} with value {value}"
                    )

        return errors

    def _get_expected_types(self) -> Dict[str, Type]:
        """
        Get expected types for configuration keys
        """
        return getattr(self, 'TYPE_HINTS', {})


class DataConfig(ConfigBase):
    """
    Configuration for data loading and preprocessing
    """

    TYPE_HINTS = {
        'train_files': (list, NoneType),
        'val_files': (list, NoneType),
        'test_files': (list, NoneType),
        'train_load_range': (list, NoneType),
        'val_load_range': (list, NoneType),
        'test_load_range': (list, NoneType),
        'train_normalized_range': (bool, NoneType),
        'val_normalized_range': (bool, NoneType),
        'test_normalized_range': (bool, NoneType),
        'tree_name': (str, NoneType),
        'inputs': dict,
        'labels': dict,
        'selection': (str, NoneType),
        'new_variables': (dict, NoneType),
        'assistants': (list, NoneType),
        'spectators': (list, NoneType),
        'weights': (dict, NoneType),
        'label_keys': list,
        'active_keys': list,
        'weight_key': str,
        'final_label_keys': list,
    }

    REQUIRED_KEYS = ['inputs', 'labels']

    def __init__(self, cfg_file_path: str):
        super().__init__(cfg_file_path)

        data = object.__getattribute__(self, '_data')
        # construct internal items
        # Per-object truth is loaded only for training, separately from the event targets.
        data['label_keys'] = flatten_nested_list([v for k, v in data['labels'].items() if k != 'objects'])
        data['active_keys'] = flatten_nested_list([
            *data['inputs'].values(),
            *data['label_keys'],
            *data.get('assistants', []),
            *data.get('weights', {}).get('vars', [])
        ])

        if data['spectators']:
            data['spectators'].extend(data['label_keys'])
        else:
            data['spectators'] = data['label_keys']

        data['weight_key'] = '_weight'
        if data['labels'].get('categorical'):
            data['final_label_keys'] = ['_class_label', *data['labels'].get('continuous', [])]
        else:
            data['final_label_keys'] = data['labels'].get('continuous', [])

        _new_var_dict = {}
        # label things
        if '_class_label' in data['final_label_keys']:
            _new_var_dict['_class_label'] = f'ak.to_numpy({data["labels"]["categorical"]}).astype(np.int64)'
            # class_labels = [f'ak.to_numpy({k})' for k in data['labels'].get('categorical')]
            # _new_var_dict['_class_label'] = f'np.argmax(np.stack([{",".join(class_labels)}], axis=1), axis=1)'
        # weight things
        # TODO: ensure type safety
        balance_weights = data.get('weights', {}).get('balance_weights', False)
        has_balance_factors = bool(data.get('weights', {}).get('weight_balance_factors'))
        has_categorical = bool(data['labels'].get('categorical'))
        match_length = False
        # if has_balance_factors and has_categorical:
        #     match_length = len(data['labels']['categorical']) == len(data['weights']['weight_balance_factors'])
        if balance_weights:
            if not all([has_balance_factors, has_categorical]):
                raise ValueError(
                    "If 'balance_weights' is True, 'weight_balance_factors' and 'categorical' labels must be provided."
                )
            # if not match_length:
            #     raise ValueError(
            #         "If 'balance_weights' is True, 'weight_balance_factors' and 'categorical' labels must have the same length."
            #     )

        _new_var_dict['_weight'] = 'ak.Array(np.ones(len(_data)))'
        if data['weights']:
            if data['weights'].get('vars'):
                if balance_weights:
                    _new_var_dict['_weight'] = '*'.join([*data['weights']['vars'], f"array({data['weights']['weight_balance_factors']})[_class_label]"])
                else:
                    _new_var_dict['_weight'] = '*'.join(data['weights']['vars'])

        if data['new_variables']:
            data['new_variables'].update(_new_var_dict)
        else:
            data['new_variables'] = _new_var_dict

    def validate(self) -> List[str]:
        """
        Validate data configuration
        """
        errors = []

        # Check required keys
        for key in self.REQUIRED_KEYS:
            if key not in self._data:
                errors.append(f"Data config missing required key: {key}")

        # Check types
        errors.extend(self._check_types())

        # Auto-set sensible defaults for missing optional values
        self._data.setdefault('train_files', None)
        self._data.setdefault('val_files', None)
        self._data.setdefault('test_files', None)
        if self._data['train_files'] is None:
            self._data['train_files'] = []
        if self._data['val_files'] is None:
            self._data['val_files'] = []
        if self._data['test_files'] is None:
            self._data['test_files'] = []
        self._data.setdefault('spectators', None)
        self._data.setdefault('assistants', None)
        self._data.setdefault('weights', None)
        self._data.setdefault('selection', None)
        self._data.setdefault('new_variables', None)

        return errors



class ModelConfig(ConfigBase):
    """
    Configuration for model architecture and hyperparameters
    """

    TYPE_HINTS = {
        'model': str,
        'model_params': dict,
        'load_model': (str, NoneType),
        'loss_function': dict,
        'training_step_function': (str, NoneType),
        'validation_step_function': (str, NoneType),
        'test_step_function': (str, NoneType),
        'start_lr': float,
        'lr_scheduler': str,
        'optimizer': str,
        'metrics': (dict, NoneType),
        'output_names': (list, NoneType),
        'binary_object_output_names': list,
        'shared_parent_outputs': dict,
        'shared_parent_veto_features': list,
        'shared_parent_output_names': list,  # Older recipes listed only the score names here.
    }

    REQUIRED_KEYS = ['model', 'model_params', 'loss_function', 'start_lr']

    def model_params_for(self, data_config):
        """Resolve shared-parent veto names against the raw object feature order used for this run."""
        params = self.model_params.copy()
        feedback = params.get("shared_parent_feedback")
        if feedback is True or (isinstance(feedback, dict) and "shared_parent_veto_features" in self):
            names = self.get("shared_parent_veto_features", DEFAULT_SHARED_PARENT_VETO)
            features = data_config.inputs["obj_feats"]
            for name in names:
                if name not in features:
                    raise ValueError(f"Shared-parent veto feature {name!r} is missing from inputs.obj_feats")
            params["shared_parent_feedback"] = {"veto_feature_indices": [features.index(name) for name in names]}
        return params

    def _upgrade_shared_parent_keys(self):
        """Read earlier Synapse-dev recipes using the current option names. ROOT branch names are values."""
        def rename(options, **old_to_new):
            if not isinstance(options, dict):
                return
            for old, new in old_to_new.items():
                if old in options:
                    if new in options:
                        raise ValueError(f"Both {old!r} and {new!r} are set. Use {new!r}.")
                    options[new] = options.pop(old)

        rename(self._data, relation_outputs="shared_parent_outputs", relation_output_names="shared_parent_output_names")
        # Model and loss parameters also accept third-party classes with their own keyword names.
        if self.get("model") == "synapse.models.ParticleTransformer.ParticleTransformerSimple":
            rename(self.get("model_params"), relation_rank="shared_parent_rank", num_relations="num_shared_parent_types",
                   relation_feedback="shared_parent_feedback")
        loss = self.get("loss_function")
        if isinstance(loss, dict) and loss.get("name") == "synapse.loss_functions.lepton_objects.EventAndLeptonObjectLoss":
            rename(loss.get("params"), relation_loss_weight="shared_parent_loss_weight",
                   relation_branches="shared_parent_branches")

    def validate(self) -> List[str]:
        """
        Validate model configuration
        """
        self._upgrade_shared_parent_keys()
        errors = []

        # Check required keys
        for key in self.REQUIRED_KEYS:
            if key not in self._data:
                errors.append(f"Model config missing required key: {key}")
        # Check types
        errors.extend(self._check_types())

        # Auto-set sensible defaults
        self._data.setdefault('lr_scheduler', None)

        veto = self.get("shared_parent_veto_features", [])
        if isinstance(veto, list) and any(not isinstance(name, str) or not name for name in veto):
            errors.append("shared_parent_veto_features must contain object feature names")

        # Validate values
        if self._data['lr_scheduler']:
            if self._data['lr_scheduler'] not in ['steps', 'cosine', 'one-cycle']:
                errors.append(f"Not supported learning rate scheduler: {self._data['lr_scheduler']}. "
                              f"Must be one of ['step', 'cosine', 'linear', 'exponential']")

        if errors:
            return errors
        loss_params = self.loss_function.get("params") or {}
        if loss_params.get("shared_parent_loss_weight") and "shared_parent_veto_features" in self:
            loss_params["shared_parent_veto_features"] = veto
        shared_parent_outputs = self.get("shared_parent_outputs")
        if shared_parent_outputs is not None:
            names = (*shared_parent_outputs.keys(), *shared_parent_outputs.values())
            if not all(isinstance(name, str) and name for name in names):
                errors.append("shared_parent_outputs must map truth branch names to score branch names")
            elif len(set(shared_parent_outputs.values())) != len(shared_parent_outputs):
                errors.append("Each shared-parent matrix must have its own score branch name")
            else:
                # The head needs only a count. The loss must read truth IDs in that same channel order.
                self.model_params["num_shared_parent_types"] = len(shared_parent_outputs)
                if loss_params.get("shared_parent_loss_weight"):
                    loss_params["shared_parent_branches"] = list(shared_parent_outputs)

        return errors


class RunConfig(ConfigBase):
    """
    Configuration for training runtime settings
    """

    TYPE_HINTS = {
        'run_dir': str,
        'run_mode': list,
        'device': str,
        'n_devices': (int, list, str),
        'load_model': (str, NoneType),
        'resume_checkpoint': (str, NoneType),
        'cosine_restart': (dict, NoneType),
        'epochs': int,
        'start_epoch': int,
        'batch_size': int,
        'num_workers': int,
        'val_sanity_check': bool,
        'cross_validation': (bool, NoneType),
        'cross_validation_var': (str, NoneType),
        'k_folds': (int, NoneType),
        'run_folds': (list, NoneType),
        'logger_config': dict,
        'seed': (int, NoneType),
        'use_amp': bool,
        'checkpoint_dir': (str, NoneType),
        'test_output': (str, NoneType),
        'binary_object_output_names': list,  # For older saved run configurations.
        'onnx_path': (str, NoneType),
        'export_onnx': (str, NoneType),
    }

    REQUIRED_KEYS = ['run_dir', 'run_mode', 'epochs', 'batch_size', 'logger_config']

    def validate(self) -> List[str]:
        """
        Validate run configuration
        """
        errors = []

        # Check required keys
        for key in self.REQUIRED_KEYS:
            if key not in self._data:
                errors.append(f"Run config missing required key: {key}")
        # Check types
        errors.extend(self._check_types())

        # Auto-set sensible defaults
        self._data.setdefault('device', 'auto')
        self._data.setdefault('n_devices', 'auto')
        self._data.setdefault('start_epoch', 0)
        self._data.setdefault('num_workers', 1)
        self._data.setdefault('val_sanity_check', False)
        self._data.setdefault('cross_validation', None)
        self._data.setdefault('cross_validation_var', None)
        self._data.setdefault('k_folds', None)
        self._data.setdefault('run_folds', None)
        self._data.setdefault('seed', None)
        self._data.setdefault('use_amp', False)
        self._data.setdefault('checkpoint_dir', None)
        self._data.setdefault('test_output', None)
        self._data.setdefault('onnx_path', None)
        self._data.setdefault('export_onnx', None)
        self._data.setdefault('load_model', None)

        # Validate values
        if self._data['epochs'] < 1:
            errors.append(f"Epochs must be >= 1, got {self._data['epochs']}")


        return errors

# TODO: consider putting model and data configs into run config
class ConfigManager:
    """
    Central manager for all configuration components
    """

    def __init__(self,
                 data_cfg_file: str,
                 model_cfg_file: str,
                 run_cfg_file: str):
        self.data = DataConfig(data_cfg_file)
        self.model = ModelConfig(model_cfg_file)
        self.run = RunConfig(run_cfg_file)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Return all configurations as nested dictionaries
        """
        return {
            'data': self.data.to_dict(),
            'model': self.model.to_dict(),
            'run': self.run.to_dict()
        }

    def save_all(self, base_path: str) -> None:
        """
        Save all configurations to directory
        """
        os.makedirs(base_path, exist_ok=True)

        self.data.save(os.path.join(base_path, 'data_config.yaml'))
        self.model.save(os.path.join(base_path, 'model_config.yaml'))
        self.run.save(os.path.join(base_path, 'run_config.yaml'))

    def merge_from_dict(self, config_dict: Dict[str, Dict[str, Any]]) -> None:
        """
        Merge configuration from a dictionary
        """
        if 'data' in config_dict:
            self.data.update(config_dict['data'])
        if 'model' in config_dict:
            self.model.update(config_dict['model'])
        if 'run' in config_dict:
            self.run.update(config_dict['run'])
