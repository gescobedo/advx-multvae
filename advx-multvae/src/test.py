import os
import shutil

import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import global_config
from src.config_classes.adv_config import AdvConfig
from src.data.user_feature import FeatureDefinition, FeatureType
from src.logging.logger import Logger
from src.modules.evaluation import AdvEval
from src.logging.utils import redirect_to_tqdm
from src.algorithms.utils import validate_epoch_test
from src.utils.input_validation import parse_input
from src.algorithms.vae import setup_model_and_optimizers_vae, preprocess_data_vae
from src.data.data_loading import get_datasets_and_loaders, get_datasets_and_loaders_mf
from src.algorithms.mf import setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf
from src.utils.training_utils import check_run_dict, run_to_fold_dict, adjust_results_dir
from src.utils.helper import reproducible, yaml_load, \
    load_model_from_path, get_device_matching_current_process, json_dump, create_unique_names, dict_apply

algorithm_fn_mapping = {
    "mf": (get_datasets_and_loaders_mf, setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf),
    "vae": (get_datasets_and_loaders, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch_test)
}


def test(algorithm, dataset, fold, model_path, results_dir, config, n_workers, devices, is_verbose=False):
    device = get_device_matching_current_process(devices)
    loading_fn, setup_fn, preprocess_fn, validation_fn = algorithm_fn_mapping[algorithm]

    if is_verbose:
        print(f"Testing '{model_path}'")

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(results_dir)

        # gather configurations of adversaries
        adv_configs = config.get("adv_groups", [])
        adv_configs = [AdvConfig(**cfg) for cfg in adv_configs]
        for cfg, name in zip(adv_configs, create_unique_names([cfg.feature for cfg in adv_configs])):
            cfg.group_name = name
        perform_adv_training = len(adv_configs) > 0

        # load data
        features = [FeatureDefinition(name=cfg.feature, type=FeatureType.from_str(cfg.type)) for cfg in adv_configs]
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold, features=features,
                                         splits=("train", "test",), run_parallel=True, n_workers=n_workers,
                                         **config["data_loader"]
                                         )
        tr_set, _ = dataset_and_loaders["train"]
        te_set, te_loader = dataset_and_loaders["test"]

        # set up model
        model, _, loss_fn, _, loss_fn_adv = setup_fn(config=config, n_users=tr_set.n_users, n_items=tr_set.n_items,
                                                     device=device, adv_configs=adv_configs)
        load_model_from_path(model, model_path, strict=False)
        model.eval()

        # set up evaluation function
        eval_fn_adv = AdvEval(adv_configs)

        # evaluate model
        results = validation_fn(model, device, te_loader, preprocess_fn, loss_fn, config, perform_adv_training,
                                loss_fn_adv, eval_fn_adv, is_verbose)
        
        # store results
        results_ind = results.copy()
        del results['metrics_individual']
        results = dict_apply(results, lambda x: x.item() if isinstance(x, torch.Tensor) else x)
        json_dump(results, os.path.join(results_dir, "results.json"))
        logger.log_value_dict("test", results, 0)

        results_ind = dict_apply(results_ind, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        pickle.dump(results_ind, open(os.path.join(results_dir, "metrics.pkl"),"wb"))

if __name__ == "__main__":

    input_config = parse_input("tester", options=["run", "experiment", "gpus", "n_folds",
                                                  "model_pattern", "n_parallel", "n_workers"],
                               access_as_properties=False)

    run_dict = input_config["run_dict"]
    if not check_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    params = []
    for fold in range(input_config["n_folds"]):
        for run_dir, run_data in fold_dict[fold].items():
            results_dirs = adjust_results_dir(run_dir, lambda _: "test")
            config = yaml_load(run_data["config_file"])

            for model_name, model_path in run_data["models"]:
                results_dir = os.path.join(results_dirs, model_name)
                # clear data from previous attacks
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir, ignore_errors=True)

                params.append({
                    "fold": fold,
                    "results_dir": results_dir,
                    "model_path": model_path,
                    "config": config
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
        delayed(test)(**arguments, **p) for p in tqdm(params, desc="Testing experiments", position=0, leave=True))
