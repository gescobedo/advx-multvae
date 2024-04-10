import os
from copy import deepcopy
from typing import Callable

import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import global_config
from src.attack import attack
from src.train import train
from src.utils.helper import get_results_dir, save_experiment_config, iter_configs, yaml_load
from src.utils.input_validation import parse_input, _populate_run_dict

from data_paths import get_data_path
from src.data.data_preparation import ensure_make_data

from src.logging.tee import Tee
from src.utils.training_utils import check_run_dict, run_to_fold_dict, adjust_results_dir

atk_fn_mapping = {
    "vae": attack,
    "mf": attack
}


def train_attack(algorithm: str, results_dir: str, dataset: str, conf: dict, fold: int, n_workers: int,
                 devices: torch.device, atk_cfg_file: dict, atk_fn: Callable, is_verbose: bool = False):
    if is_verbose:
        print(f"Training with '{conf_name}'")

    train(algorithm, results_dir, dataset, conf, fold, n_workers, devices, is_verbose=is_verbose)

    if is_verbose:
        print(f"Training done, attacking model(s)")

    run_dict = {}
    _populate_run_dict(results_dir, run_dict, "*.pt*")

    if not check_run_dict(run_dict, is_verbose):
        exit()
    fold_dict = run_to_fold_dict(run_dict, is_verbose)

    for attacker_config, attacker_name in iter_configs(atk_cfg_file):
        for run_dir, run_data in fold_dict[fold].items():
            result_dirs = adjust_results_dir(run_dir, lambda _: "atk")
            result_dirs = os.path.join(result_dirs, attacker_name)
            pretrained_config = yaml_load(run_data["config_file"])

            for model_name, model_path in run_data["models"]:
                if is_verbose:
                    print(f"Attacking '{conf_name}' with '{model_name}'")
                result_dir = os.path.join(result_dirs, model_name)
                atk_fn(algorithm, dataset, fold, model_path, result_dir, deepcopy(pretrained_config),
                       deepcopy(attacker_config), n_workers, devices, is_verbose=is_verbose)


if __name__ == "__main__":
    with Tee() as tee:
        tee.write_cmd_line_call()
        input_config = parse_input("experiments",
                                   options=["gpus", "algorithm", "config", "atk_config", "dataset", "n_folds",
                                            "n_workers", "n_parallel", "resampling_strategy", "features"],
                                   access_as_properties=False)
        results_dir = get_results_dir(input_config["dataset"], input_config["algorithm"])
        os.makedirs(results_dir, exist_ok=False)
        tee.set_file(os.path.join(results_dir, "output.log"))
        save_experiment_config(input_config["config"], results_dir)

        # Ensure that dataset exists now, as otherwise this may be called multiple times
        # when running experiments in parallel
        data_path = get_data_path(input_config["dataset"])
        ensure_make_data(data_path, n_folds=global_config.MAX_FOLDS, target_path=data_path,
                         resampling_strategy=input_config["resampling_strategy"],
                         features=input_config["features"],
                         random_state=global_config.EXP_SEED)

        params = []
        for fold in range(input_config["n_folds"]):
            for conf, conf_name in iter_configs(input_config["config"]):
                params.append({
                    "fold": fold,
                    "conf": conf,
                    "results_dir": os.path.join(results_dir, str(fold), "train", conf_name)
                })

        devices = input_config["devices"]
        n_devices = len(devices)
        is_verbose = len(params) == 1 or (n_devices == 1 and input_config["n_parallel"] == 1)

        arguments = {
            "dataset": input_config["dataset"],
            "n_workers": input_config["n_workers"],
            "algorithm": input_config["algorithm"],
            "devices": devices,
            "is_verbose": is_verbose,
            "atk_cfg_file": input_config["atk_config"],
            "atk_fn": atk_fn_mapping[input_config["algorithm"]]
        }

        print("Running", len(params), "training job(s)")
        Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
            delayed(train_attack)(**arguments, **p) for p in tqdm(params, desc="Running experiments",
                                                                  position=0, leave=True))
        print("Done.")
