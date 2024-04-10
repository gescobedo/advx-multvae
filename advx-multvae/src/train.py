import os

import numpy as np
import torch
from joblib import Parallel, delayed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm

import global_config
from src.config_classes.adv_config import AdvConfig
from src.data.user_feature import FeatureDefinition, FeatureType
from src.logging.logger import Logger
from src.logging.utils import redirect_to_tqdm
from src.modules.evaluation import AdvEval
from src.utils.helper import get_results_dir, save_experiment_config, iter_configs, get_device_matching_current_process, \
    reproducible, yaml_dump, save_model, create_unique_names
from src.utils.input_validation import parse_input

from data_paths import get_data_path
from src.data.data_preparation import ensure_make_data
from src.data.data_loading import get_datasets_and_loaders_mf, get_datasets_and_loaders
from src.algorithms.utils import validate_epoch, train_epoch_parallel
from src.algorithms.mf import setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf
from src.algorithms.vae import preprocess_data_vae, setup_model_and_optimizers_vae
from src.logging.tee import Tee

algorithm_fn_mapping = {
    "mf": (get_datasets_and_loaders_mf, setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf),
    "vae": (get_datasets_and_loaders, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch)
}


def train(algorithm: str, results_dir: str, dataset: str, conf: dict, fold: int,
          n_workers: int, devices: torch.device, is_verbose: bool = False):
    device = get_device_matching_current_process(devices)

    loading_fn, setup_fn, preprocess_fn, validation_fn = algorithm_fn_mapping[algorithm]

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(results_dir)

        # Store used config for later retrieval
        yaml_dump(conf, os.path.join(results_dir, "config.yaml"))
        perform_adv_training = len(conf["adv_groups"]) > 0

        # gather configuration for adversaries
        adv_configs = conf.get("adv_groups", [])
        adv_configs = [AdvConfig(**config) for config in adv_configs]
        for cfg, name in zip(adv_configs, create_unique_names([cfg.feature for cfg in adv_configs])):
            cfg.group_name = name

        # Load data
        features = [FeatureDefinition(name=cfg.feature, type=FeatureType.from_str(cfg.type)) for cfg in adv_configs]
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold, features=features,
                                         splits=("train", "val"), run_parallel=True, n_workers=n_workers,
                                         **conf["data_loader"]
                                         )

        tr_set, tr_loader = dataset_and_loaders["train"]
        vd_set, vd_loader = dataset_and_loaders["val"]

        model, optim, loss_fn, optim_adv, loss_fn_adv = setup_fn(conf, tr_set.n_users, tr_set.n_items,
                                                                 device, adv_configs)
        # also define evaluation function.
        # note that we don't do that in the setup_fn as evaluation should be separate from training
        eval_fn_adv = AdvEval(adv_configs)

        early_stopping_criteria = conf.get("early_stopping_criteria")
        if early_stopping_criteria is None:
            early_stopping_criteria = {}

        if scheduler := conf.get("scheduler"):
            scheduler = ReduceLROnPlateau(optim, **scheduler)
        if scheduler_adv := conf.get("adv_scheduler"):
            scheduler_adv = ReduceLROnPlateau(optim_adv, **scheduler_adv)

        # any "inf" value works for a "closest_is_best" criterion
        best_validation_scores = {label: (-np.Inf if crit.get("highest_is_best") else np.Inf)
                                  for label, crit in early_stopping_criteria.items()}

        store_model_every = conf.get("store_model_every", conf["epochs"])

        for epoch in trange(conf["epochs"], desc="Epochs", position=1, leave=True, disable=not is_verbose):
            train_epoch_parallel(model, optim, device, tr_loader, preprocess_fn, loss_fn, logger, epoch,
                                 perform_adv_training, optim_adv, loss_fn_adv,
                                 log_every_n_batches=conf["logging"]["log_every_n_batches"],
                                 is_verbose=is_verbose)

            best_validation_scores = validation_fn(model, device, vd_loader, preprocess_fn, loss_fn, logger, epoch,
                                                   results_dir, best_validation_scores, early_stopping_criteria, conf,
                                                   perform_adv_training, loss_fn_adv, eval_fn_adv, is_verbose,
                                                   tr_set=tr_set, vd_set=vd_set,
                                                   scheduler=scheduler, scheduler_adv=scheduler_adv)
            logger.store()

            if epoch % store_model_every == 0 and epoch > 0:
                save_model(model, results_dir, f"model_epoch_{epoch}")

        yaml_dump(best_validation_scores, os.path.join(results_dir, "best_validation_scores.json"))
        if conf["store_last_model"]:
            save_model(model, results_dir, f"model_epoch_{epoch}")


if __name__ == "__main__":
    with Tee() as tee:
        tee.write_cmd_line_call()
        input_config = parse_input("experiments",
                                   options=["gpus", "algorithm", "config", "dataset", "n_folds",
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
            "is_verbose": is_verbose
        }

        print("Running", len(params), "training job(s)")
        Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
            delayed(train)(**arguments, **p) for p in tqdm(params, desc="Running experiments", position=0, leave=True))
        print("Done.")
