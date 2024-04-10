import multiprocessing
import os
import re
import math
import json
import pickle
import random
import shutil
from typing import Callable

import numpy as np
import scipy.sparse.csc
from copy import deepcopy
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import ParameterGrid

import torch
from torch import nn

from data_paths import get_storage_path

# Using ruamel.yaml instead of yaml to be able to parse scientific numbers
# cf https://yaml.readthedocs.io/en/latest/pyyaml.html
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

ARRAY_DICT_KEY_PATTERN = r"^((([a-zA-Z]+\w*)\[(\d+)\])|([a-zA-Z]+\w*))$"


def reproducible(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pickle_load(file_path):
    with open(file_path, 'rb') as fh:
        return pickle.load(fh)


def pickle_dump(obj, file_path):
    with open(file_path, 'wb') as fh:
        return pickle.dump(obj, fh)


def json_load(file_path):
    with open(file_path, 'r') as fh:
        return json.load(fh)


def json_dump(obj, file_path):
    with open(file_path, 'w') as fh:
        return json.dump(obj, fh, indent=4)


def yaml_load(file_path):
    with open(file_path, 'r') as fh:
        return yaml.load(fh)


def yaml_dump(obj, file_path):
    with open(file_path, 'w') as fh:
        return yaml.dump(obj, fh)


def replace_special_path_chars(path):
    # replace any sort of characters that don't work on either Windows or Linux
    # special chars from https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
    special_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    path = path.replace(": ", '=')
    for c in special_chars:
        path = path.replace(c, '_')

    return path


def modularize_config(conf: dict, separator: str = ".") -> dict:
    """
    Brings a previously flattened dictionary, i.e., where keys to values are flattened,
    back to its original module structure.

    For example, the dictionary
    {
        "foo.bar.key1": 1,
        "foo.key2": 2
    }
    leads to
    {
        "foo": {
            "bar": {"key1": 1},
            "key2": 2
        }
    }

    Parameters:
    :param conf: The config dict to modularize
    :param separator: The separator to use for
    :return: The modularized (un-flattened) dictionary.
    """
    sc = defaultdict(lambda: dict())
    for k, v in conf.items():
        prefix, name = k.split(separator, maxsplit=1)
        sc[prefix][name] = v
    return dict(sc)


def flatten_config(conf: dict):
    flattened_conf = dict()
    for k, v in conf.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                flattened_conf[k + "|" + k1] = v1
    return flattened_conf


def get_results_dir(dataset_name, experiment_type):
    now = datetime.now()
    results_base_path = get_storage_path()
    results_path = os.path.join(results_base_path,
                                dataset_name,
                                f"{experiment_type}--{now.strftime('%Y-%m-%d_%H-%M-%S')}")
    print(f"Storing results in '{results_path}'")
    return results_path


def pretty_print(d: dict):
    print(json.dumps(d, indent="  "))


def __save_len(a):
    if isinstance(a, scipy.sparse.spmatrix):
        # Scipy sparse matrices do not support __len__()
        return a.shape[0]
    return len(a)


def chunkify(a, chunk_size):
    assert chunk_size > 0, "Chunk size must be > 0"
    chunk_size = int(chunk_size)  # prevent missing cast to int
    n = int(math.ceil(__save_len(a) / chunk_size))
    return (a[i * chunk_size: i * chunk_size + chunk_size] for i in range(n))


def split(a, n):
    chunk_size = math.ceil(__save_len(a) / n)
    return chunkify(a, chunk_size)


def chunkify_multi(col, chunk_size):
    return zip(*[chunkify(a, chunk_size) for a in col])


def __is_model_on_cpu(model):
    return __get_model_device(model).type == "cpu"


def __get_model_device(model):
    # Assuming we do not distribute model over different devices
    return next(model.parameters()).device


def save_model(model, dir, name):
    """
    Utility to save a models state dict, no matter whether it is on CPU or GPU.
    Moreover, we can ensure only a single file extension.
    """
    if not __is_model_on_cpu(model):
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        state_dict = model.state_dict()

    os.makedirs(dir, exist_ok=True)
    torch.save(state_dict, os.path.join(dir, name + ".pt"))


def load_model(model, dir, name):
    """ Utility function to load a model, ensuring that only one kind of file extension is used."""
    return load_model_from_path(model, os.path.join(dir, name + ".pt"))


def load_model_from_path(model: nn.Module, path: str, strict: bool = True):
    device = __get_model_device(model)
    return model.load_state_dict(torch.load(path, map_location=device), strict=strict)


def merge_dicts(first: dict, second: dict):
    """
    Merges two dictionaries and all their subsequent dictionaries.
    In case both dictionaries contain the same key, which is not another dictionary, the latter one is used.

    This merges in contrast to dict.update() all subdicts and its items
    instead of overriding the former with the latter.
    """
    fk = set(first.keys())
    sk = set(second.keys())
    common_keys = fk.intersection(sk)

    z = {}
    for k in common_keys:
        if isinstance(first[k], dict) and isinstance(second[k], dict):
            z[k] = merge_dicts(first[k], second[k])
        else:
            z[k] = deepcopy(second[k])

    for k in fk - common_keys:
        z[k] = deepcopy(first[k])

    for k in sk - common_keys:
        z[k] = deepcopy(second[k])

    return z


def iter_configs(config_path):
    config = yaml_load(config_path)

    default_config = {}
    if extended_config_file := config.get("extends"):
        extended_config_file = os.path.join(os.path.dirname(config_path), extended_config_file)
        default_config = yaml_load(extended_config_file)

    whole_config = default_config
    if fixed_params := config.get("fixed_params"):
        whole_config = merge_dicts(default_config, fixed_params)

    grid_params = config.get("grid_params")
    if grid_params is None or len(grid_params.keys()) == 0:
        yield whole_config, "run"
        return

    for pg_config in ParameterGrid(grid_params):
        cfg = deepcopy(whole_config)
        for k, v in pg_config.items():
            nested_dict_set(k, v, cfg)
        yield cfg, replace_special_path_chars(str(pg_config))


def save_experiment_config(config_path, destination):
    ext = os.path.splitext(config_path)[1]
    os.makedirs(destination, exist_ok=True)

    config = yaml_load(config_path)
    if extended_config_file := config.get("extends"):
        extended_config_file = os.path.join(os.path.dirname(config_path), extended_config_file)
        default_file_name = "default_config" + ext
        shutil.copyfile(extended_config_file, os.path.join(destination, default_file_name))
        # maintain "extends" property
        config["extends"] = default_file_name

    yaml_dump(config, os.path.join(destination, "config" + ext))


def get_device_matching_current_process(devices):
    identity = multiprocessing.current_process()._identity
    process_id = 0 if len(identity) == 0 else (identity[0] - 1)

    if process_id >= len(devices):
        print(f"Error! PID of parallel process ({process_id}) greater than number of devices ({len(devices)}). "
              f"Using PID {process_id % len(devices)} instead.")
        process_id %= len(devices)

    device = devices[process_id]
    print(f"Process {process_id} running on device", device)
    return device


def nested_dict_set(key: str, value: any, d: dict, sep: str = ".") -> dict:
    """
    Iterates through the provided nested dictionary and searches for the provided key,
    and then sets its value to the one specified. As the dictionary might consist
    of other dictionaries, a separator (sep) can be used to access
    sub-dictionaries.

    Note that as dictionaries are mutable and passed by reference, we modify the dictionary in-place
    to make the requested changes. Still, the dict is returned to suit all use-cases.

    Args:
        key (str): the key to look for in the dictionary
        value (any): the value to set the dictionary item to
        d (dict): the dictionary for which key values should be replaced
        sep (str): the string/char that splits the different levels of the dict keys

    Returns:
        (dict): the modified dictionary (Note that the dictionary is modified in-place nonetheless.)
    """
    return _nested_dict_set(key, key, value, d, sep)


def _split_with_rest(s: str, sep: str, n_splits=1):
    """
    splits the specified string, if string was split less than "n_splits" times,
    for the remaining splits, None will be returned
    """
    result = s.split(sep, maxsplit=n_splits)
    return tuple(list(result) + ([None] * (n_splits + 1 - len(result))))


def _nested_dict_set(full_key: str, key: str, value: any, d: dict, sep: str = ".") -> dict:
    di = d
    k, rest = _split_with_rest(key, sep, n_splits=1)

    # basically validates and searches for list indices
    res = re.search(ARRAY_DICT_KEY_PATTERN, k)
    if res is None:
        raise KeyError(f"format of key '{k}' not supported.")
    _, _, key_for_list, list_idx, key_for_dict = res.groups()
    list_idx_provided = list_idx is not None

    k = key_for_list or key_for_dict
    list_idx = int(list_idx) if list_idx_provided else None

    # look up whether key exists in dict
    if isinstance(di, dict) and k in di:
        if list_idx_provided:
            if not isinstance(di[k], list):
                raise KeyError(f"list index provided with key {full_key} but no list found.")

        if rest is None:
            if list_idx_provided:
                di[k][list_idx] = value
            else:
                di[k] = value
            return di

        # continue search
        return _nested_dict_set(full_key, rest, value, di[k][list_idx] if list_idx_provided else di[k])

    else:
        # failed to find key
        raise KeyError(full_key)


def create_unique_names(names: list):
    counter = Counter(names)
    non_unique_names = [n for n, c in counter.items() if c >= 2]
    non_unique_counter = {n: 0 for n in non_unique_names}

    unique_names = []
    for n in names:
        if n in non_unique_names:
            unique_names.append(f"{n}_{non_unique_counter[n]}")
            non_unique_counter[n] += 1
        else:
            unique_names.append(n)

    return unique_names


def dict_apply(d: dict, fn: Callable):
    for k, v in d.items():
        result = dict_apply(v, fn) if isinstance(v, dict) else fn(v)
        d[k] = result
    return d

