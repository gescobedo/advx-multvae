import os
import shutil
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed

import torch
from torch.optim import Adam

import global_config
import src.modules.parallel
from src.config_classes.atk_config import AtkConfig
from src.data.user_feature import FeatureDefinition, FeatureType
from src.logging.logger import Logger
from src.modules.evaluation import AdvEval
from src.modules.losses import AdvLosses
from src.modules.polylinear import PolyLinearParallel
from src.logging.utils import redirect_to_tqdm
from src.algorithms.utils import validate_epoch, postprocess_loss_results, add_to_dict, scale_dict_items
from src.utils.input_validation import parse_input
from src.algorithms.vae import setup_model_and_optimizers_vae, preprocess_data_vae
from src.data.data_loading import get_datasets_and_loaders, get_datasets_and_loaders_mf
from src.algorithms.mf import setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf
from src.utils.training_utils import check_run_dict, run_to_fold_dict, adjust_results_dir
from src.utils.helper import iter_configs, reproducible, yaml_load, \
    load_model_from_path, get_device_matching_current_process, json_dump, pickle_dump, create_unique_names

algorithm_fn_mapping = {
    "mf": (get_datasets_and_loaders_mf, setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf),
    "vae": (get_datasets_and_loaders, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch)
}


def attack(algorithm, dataset, fold, model_path, results_dir, pretrained_config, attacker_config,
           n_workers, devices, is_verbose=False):
    device = get_device_matching_current_process(devices)
    loading_fn, setup_fn, preprocess_fn, validation_fn = algorithm_fn_mapping[algorithm]

    if is_verbose:
        print("Attacking", model_path)

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(results_dir)

        # gather configuration for attackers
        atk_configs = attacker_config.get("atk_groups", [])
        atk_configs = [AtkConfig(**config) for config in atk_configs]
        for cfg, name in zip(atk_configs, create_unique_names([cfg.feature for cfg in atk_configs])):
            cfg.group_name = name

        # load data
        features = [FeatureDefinition(name=cfg.feature, type=FeatureType.from_str(cfg.type)) for cfg in atk_configs]
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold, features=features,
                                         splits=("train", "val", "test"), run_parallel=True, n_workers=n_workers,
                                         **pretrained_config["data_loader"]
                                         )
        tr_set, tr_loader = dataset_and_loaders["train"]
        vd_set, vd_loader = dataset_and_loaders["val"]
        te_set, te_loader = dataset_and_loaders["test"]

        # set adv_configs to 'None' to disable adversaries
        pretrained_model, *_ = setup_fn(config=pretrained_config, n_users=tr_set.n_users, n_items=tr_set.n_items,
                                        device=device, adv_configs=None)
        load_model_from_path(pretrained_model, model_path, strict=False)
        pretrained_model.eval()

        attacker, optim, loss_fn, eval_fn = setup_attacker(input_size=pretrained_model.get_encoding_size(),
                                                           full_config=attacker_config, atk_configs=atk_configs,
                                                           device=device)

        for epoch in trange(attacker_config["epochs"], desc="Epochs", position=1, leave=True, disable=not is_verbose):
            # train attacker
            attacker.train()
            run_epoch("train", attacker, pretrained_model, preprocess_fn, loss_fn, eval_fn, tr_loader,
                      device, epoch, logger, optim=optim, return_raw_results=False)

            # validate attacker
            attacker.eval()
            with torch.no_grad():
                run_epoch("val", attacker, pretrained_model, preprocess_fn, loss_fn, eval_fn, vd_loader,
                          device, epoch, logger, optim=None, return_raw_results=False)

        # test attacker
        attacker.eval()
        with torch.no_grad():
            eval_dict, raw_results = run_epoch("test", attacker, pretrained_model, preprocess_fn, loss_fn, eval_fn,
                                               te_loader, device, epoch, logger, optim=None, return_raw_results=True)

        # Store attacker data to be able to analyze its performance later on
        pickle_dump({
            **raw_results,
            "user_feature_map": {k: v.value_map for k, v in te_set.user_features.items() if v.is_categorical_feature}
        }, os.path.join(results_dir, f"test_set_attacker_data.pkl"))

        # some weird yaml error prevents saving the dict, therefore use json for now
        json_dump(eval_dict, os.path.join(results_dir, f"test_set_attacker_evaluation.json"))


def run_epoch(split, attacker, pretrained_model, preprocess_fn, loss_fn, eval_fn, data_loader, device, epoch, logger,
              optim=None, return_raw_results=False):
    aggregated_loss_dict = defaultdict(lambda: 0)
    aggregated_eval_dict = defaultdict(lambda: 0)
    sample_count, aggregated_loss = 0, 0.

    raw_data = defaultdict(lambda: list())

    for indices, *model_input, _, targets in tqdm(data_loader, desc="Steps", position=2, leave=True, disable=True):
        n_samples = len(indices)
        sample_count += n_samples

        targets = [t.to(device) for t in targets]
        model_input = preprocess_fn(model_input, device)

        with torch.no_grad():
            latent_user = pretrained_model.encode_user(*model_input)
        logits = attacker(latent_user)
        result = loss_fn(logits, targets)

        loss, loss_dict = postprocess_loss_results(result)
        eval_dict = eval_fn(logits, targets)

        if optim is not None:
            # Update model
            optim.zero_grad()
            loss.backward()
            optim.step()

        # store results over batches
        aggregated_loss += loss * n_samples
        aggregated_loss_dict = add_to_dict(aggregated_loss_dict, loss_dict, multiplier=n_samples)
        aggregated_eval_dict = add_to_dict(aggregated_eval_dict, eval_dict, multiplier=n_samples)

        if return_raw_results:
            raw_data["indices"].append(indices)
            raw_data["logits"].append([[log.detach().cpu() for log in log_grp] for log_grp in logits])
            raw_data["targets"].append([tar.detach().cpu() for tar in targets])

    logger.log_value(f"{split}/atk_loss", aggregated_loss / sample_count, epoch)
    logger.log_value_dict(f"{split}/atk_loss", scale_dict_items(aggregated_loss_dict, 1 / sample_count), epoch)

    aggregated_eval_dict = scale_dict_items(aggregated_eval_dict, 1 / sample_count)
    logger.log_value_dict(f"{split}/atk_eval", aggregated_eval_dict, epoch)

    if return_raw_results:
        raw_data = dict(raw_data)
        raw_data["indices"] = np.concatenate(raw_data["indices"])
        raw_data["logits"] = [[np.concatenate(log) for log in zip(*grp)] for grp in zip(*raw_data["logits"])]
        raw_data["targets"] = [np.concatenate(grp) for grp in zip(*raw_data["targets"])]
        return aggregated_eval_dict, raw_data


def setup_attacker(input_size: int, full_config: dict, atk_configs: List[AtkConfig], device: torch.device):
    attacker_modules = [
        PolyLinearParallel(layer_config=[input_size] + config.dims,
                           n_parallel=config.n_parallel,
                           input_dropout=config.input_dropout,
                           activation_fn=config.activation_fn
                           )
        for config in atk_configs
    ]

    attacker = src.modules.parallel.Parallel(modules=attacker_modules,
                                             parallel_mode=src.modules.parallel.ParallelMode.SingleInMultiOut)
    attacker.to(device)

    loss_fn = AdvLosses(atk_configs)
    eval_fn = AdvEval(atk_configs)

    optim = Adam(attacker.parameters(), **full_config["optim"])

    return attacker, optim, loss_fn, eval_fn


if __name__ == "__main__":

    input_config = parse_input("attacker", options=["run", "experiment", "gpus", "config",
                                                    "n_folds", "model_pattern", "n_parallel", "n_workers"],
                               access_as_properties=False)

    run_dict = input_config["run_dict"]
    if not check_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    params = []
    for fold in range(input_config["n_folds"]):
        for attacker_config, attacker_name in iter_configs(input_config["config"]):
            for run_dir, run_data in fold_dict[fold].items():
                results_dirs = adjust_results_dir(run_dir, lambda _: "atk")
                results_dirs = os.path.join(results_dirs, attacker_name)
                pretrained_config = yaml_load(run_data["config_file"])

                for model_name, model_path in run_data["models"]:
                    results_dir = os.path.join(results_dirs, model_name)
                    # clear data from previous attacks
                    if os.path.exists(results_dir):
                        shutil.rmtree(results_dir, ignore_errors=True)

                    params.append({
                        "fold": fold,
                        "results_dir": results_dir,
                        "model_path": model_path,
                        "attacker_config": attacker_config,
                        "pretrained_config": pretrained_config
                    })

    devices = input_config["devices"]
    n_devices = len(devices)
    is_verbose = len(params) == 1 or (n_devices == 1 and input_config["n_parallel"] == 1)

    arguments = {
        "dataset": input_config["dataset"],
        "n_workers": input_config["n_workers"],
        "algorithm": input_config["algorithm"],
        "devices": input_config["devices"],
        "is_verbose": is_verbose
    }

    print("Running", len(params), "training jobs")
    Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
        delayed(attack)(**arguments, **p) for p in tqdm(params, desc="Attacking experiments", position=0, leave=True))
