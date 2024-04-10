import os
import shutil
from collections import defaultdict

from rmet import calculate
from tqdm import tqdm
from joblib import Parallel, delayed

import torch

import global_config
from src.logging.logger import Logger
from src.utils.adv_evaluation import eval_adversaries, eval_adversaries_cont
from src.logging.utils import redirect_to_tqdm
from src.algorithms.utils import validate_epoch
from src.utils.input_validation import parse_input
from src.algorithms.vae import setup_model_and_optimizers_vae, preprocess_data_vae
from src.data.data_loading import get_datasets_and_loaders, get_datasets_and_loaders_mf, get_datasets_and_loaders_cont
from src.algorithms.mf import setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf
from src.utils.training_utils import check_run_dict, run_to_fold_dict, adjust_results_dir
from src.utils.helper import reproducible, yaml_dump, yaml_load, \
    load_model_from_path, get_device_matching_current_process, pickle_dump

algorithm_fn_mapping = {
    "mf": (get_datasets_and_loaders_mf, setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf,
           eval_adversaries),
    "vae": (
        get_datasets_and_loaders, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch,
        eval_adversaries),
    "vae-cont": (get_datasets_and_loaders_cont, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch,
                 eval_adversaries_cont),
}

TOP_K_GATHER = 200


def gather(algorithm, dataset, split, fold, model_path, experiment_config, log_dir, n_workers, devices,
           is_verbose=False):
    device = get_device_matching_current_process(devices)
    loading_fn, setup_fn, preprocess_fn, validation_fn, adv_eval_fn = algorithm_fn_mapping[algorithm]

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(log_dir)

        # Load data
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold,
                                         splits=("train", split), run_parallel=True, n_workers=n_workers,
                                         **experiment_config["data_loader"]
                                         )
        tr_set, tr_loader = dataset_and_loaders["train"]
        eval_set, eval_loader = dataset_and_loaders[split]

        model, *_, loss_fn, loss_fn_adv = setup_fn(experiment_config, tr_set.n_users, tr_set.n_items, device)
        load_model_from_path(model, model_path)
        model.eval()

        top_k = experiment_config["top_k"]
        perform_adv_training = experiment_config["adv"]["perform_adv_training"]

        sample_count = 0
        bacc = 0
        metric_values = defaultdict(lambda: defaultdict(lambda: 0))
        metrics_data = defaultdict(lambda: defaultdict(lambda: []))
        gathered_data = defaultdict(lambda: [])
        with torch.no_grad():
            for indices, *model_input, targets, adv_targets in tqdm(eval_loader,
                                                                    desc=f"Gathering data of '{split}' split",
                                                                    position=1, leave=True, disable=True):
                n_samples = len(adv_targets)
                sample_count += n_samples
                targets = targets.to(device)

                adv_targets = adv_targets.flatten()
                adv_targets = adv_targets.to(device, dtype=torch.long)
                model_input = preprocess_fn(model_input, device)

                latent_user = model.encode_user(*model_input)
                logits, loss_dict, adv_logits = model(*model_input)

                # Removing items from input data
                logits[model_input[0].nonzero(as_tuple=True)] = .0

                if perform_adv_training:
                    _, adv_bacc = adv_eval_fn(adv_logits, adv_targets, loss_fn_adv)
                    bacc += adv_bacc.cpu().item() * n_samples

                # Gather metrics, aggregated over the whole user base
                metric_results = calculate(experiment_config["metrics"], logits, targets, top_k,
                                           return_aggregated=True, return_individual=False)
                for k, v in metric_results.items():
                    for t, v1 in v.items():
                        metric_values[k][t] += v1 * n_samples

                # Gather metrics of individual users
                # Note: For simplicity we calculate the metrics twice (computational costs are low)
                metric_results = calculate(experiment_config["metrics"], logits, targets, top_k,
                                           return_aggregated=False, return_individual=True)
                for k, v in metric_results.items():
                    for t, v1 in v.items():
                        metrics_data[k][t].append(v1)

                gathered_data["indices"].append(indices)
                gathered_data["n_interactions"].append(torch.sum(targets, dim=-1).cpu())
                gathered_data["output"].append(torch.topk(logits, TOP_K_GATHER, dim=-1).indices.cpu())

                # TODO: this is not quite correct: we want to gather all targets (to have precise results),
                #       not just some of them. This is especially the case as only the earlier 'TOP_K_GATHER'
                #       items would be selected...
                gathered_data["targets"].append(torch.topk(targets, TOP_K_GATHER, dim=-1).indices.cpu())

                gathered_data["latent"].append(latent_user)

                if perform_adv_training:
                    gathered_data["adv_targets"].append(adv_targets.cpu())

        metric_values = {k: {t: v1 / sample_count for t, v1 in v.items()} for k, v in metric_values.items()}
        metric_values["adv_bacc"] = bacc / sample_count
        yaml_dump(metric_values, os.path.join(log_dir, "metric_values.yaml"))

        metrics_data = {k: {t: torch.concat(v1).cpu().numpy() for t, v1 in v.items() if isinstance(v1[0], torch.Tensor)}
                        for k, v in metrics_data.items()}
        pickle_dump(metrics_data, os.path.join(log_dir, "metrics_data.pkl"))

        gathered_data = {k: torch.concat(v).cpu().numpy() for k, v in gathered_data.items()}
        gathered_data["user_feature_map"] = {k: v.value_map for k, v in eval_set.user_features.items()}
        pickle_dump(gathered_data, os.path.join(log_dir, "gathered_data.pkl"))

        # May be nice to make results available in TensorBoard as well
        for k, v in metric_values.items():
            if isinstance(v, dict):
                for t, v1 in v.items():
                    logger.log_value(f"{split}/{k}@{t}", v1, 0)
            else:
                logger.log_value(f"{split}/{k}", v, 0)


def gather_cont(algorithm, dataset, split, fold, model_path, experiment_config, log_dir, n_workers, devices,
                is_verbose=False):
    device = get_device_matching_current_process(devices)
    loading_fn, setup_fn, preprocess_fn, validation_fn, adv_eval_fn = algorithm_fn_mapping[algorithm]

    with redirect_to_tqdm():
        reproducible(global_config.EXP_SEED)
        logger = Logger(log_dir)

        # Load data
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold,
                                         splits=("train", split), run_parallel=True, n_workers=n_workers,
                                         **experiment_config["data_loader"]
                                         )
        tr_set, tr_loader = dataset_and_loaders["train"]
        eval_set, eval_loader = dataset_and_loaders[split]

        model, *_, loss_fn, loss_fn_adv = setup_fn(experiment_config, tr_set.n_users, tr_set.n_items, device)
        load_model_from_path(model, model_path)
        model.eval()

        top_k = experiment_config["top_k"]
        perform_adv_training = experiment_config["adv"]["perform_adv_training"]

        sample_count = 0
        bacc = 0
        metric_values = defaultdict(lambda: defaultdict(lambda: 0))
        metrics_data = defaultdict(lambda: defaultdict(lambda: []))
        gathered_data = defaultdict(lambda: [])
        with torch.no_grad():
            for indices, *model_input, targets, adv_targets in tqdm(eval_loader,
                                                                    desc=f"Gathering data of '{split}' split",
                                                                    position=1, leave=True, disable=True):
                n_samples = len(adv_targets)
                sample_count += n_samples
                targets = targets.to(device)

                # adv_targets = adv_targets
                adv_targets = adv_targets.to(device, dtype=torch.float)
                model_input = preprocess_fn(model_input, device)

                latent_user = model.encode_user(*model_input)
                logits, loss_dict, adv_logits = model(*model_input)

                # Removing items from input data
                logits[model_input[0].nonzero(as_tuple=True)] = .0

                if perform_adv_training:
                    _, adv_bacc = adv_eval_fn(adv_logits, adv_targets, loss_fn_adv)
                    bacc += adv_bacc.cpu().item() * n_samples

                # Gather metrics, aggregated over the whole user base
                metric_results = calculate(experiment_config["metrics"], logits, targets, top_k,
                                           return_aggregated=True, return_individual=False)
                for k, v in metric_results.items():
                    for t, v1 in v.items():
                        metric_values[k][t] += v1 * n_samples

                # Gather metrics of individual users
                # Note: For simplicity we calculate the metrics twice (computational costs are low)
                metric_results = calculate(experiment_config["metrics"], logits, targets, top_k,
                                           return_aggregated=False, return_individual=True)
                for k, v in metric_results.items():
                    for t, v1 in v.items():
                        metrics_data[k][t].append(v1)

                gathered_data["indices"].append(indices)
                gathered_data["n_interactions"].append(torch.sum(targets, dim=-1).cpu())
                gathered_data["output"].append(torch.topk(logits, TOP_K_GATHER, dim=-1).indices.cpu())

                # TODO: this is not quite correct: we want to gather all targets (to have precise results),
                #       not just some of them. This is especially the case as only the earlier 'TOP_K_GATHER'
                #       items would be selected...
                gathered_data["targets"].append(torch.topk(targets, TOP_K_GATHER, dim=-1).indices.cpu())

                gathered_data["latent"].append(latent_user)

                if perform_adv_training:
                    gathered_data["adv_targets"].append(adv_targets.cpu())

        metric_values = {k: {t: v1 / sample_count for t, v1 in v.items()} for k, v in metric_values.items()}
        metric_values["adv_bacc"] = bacc / sample_count
        yaml_dump(metric_values, os.path.join(log_dir, "metric_values.yaml"))

        metrics_data = {k: {t: torch.concat(v1).cpu().numpy() for t, v1 in v.items() if isinstance(v1[0], torch.Tensor)}
                        for k, v in metrics_data.items()}
        pickle_dump(metrics_data, os.path.join(log_dir, "metrics_data.pkl"))

        gathered_data = {k: torch.concat(v).cpu().numpy() for k, v in gathered_data.items()}
        gathered_data["user_feature_map"] = {k: v.value_map for k, v in eval_set.user_features.items()}
        pickle_dump(gathered_data, os.path.join(log_dir, "gathered_data.pkl"))

        # May be nice to make results available in TensorBoard as well
        for k, v in metric_values.items():
            if isinstance(v, dict):
                for t, v1 in v.items():
                    logger.log_value(f"{split}/{k}@{t}", v1, 0)
            else:
                logger.log_value(f"{split}/{k}", v, 0)


if __name__ == "__main__":

    input_config = parse_input("gather", options=["run", "experiment", "gpus", "split",
                                                  "n_folds", "model_pattern", "n_parallel", "n_workers"],
                               access_as_properties=False)

    run_dict = input_config["run_dict"]
    if not check_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    params = []
    for fold in range(input_config["n_folds"]):
        for run_dir, run_data in fold_dict[fold].items():
            result_dirs = adjust_results_dir(run_dir, lambda _: "gather")
            model_config = yaml_load(run_data["config_file"])

            for model_name, model_path in run_data["models"]:
                result_dir = os.path.join(result_dirs, model_name)
                # clear data from previous run
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir, ignore_errors=True)

                params.append({
                    "fold": fold,
                    "log_dir": result_dir,
                    "model_path": model_path,
                    "experiment_config": model_config
                })

    devices = input_config["devices"]
    n_devices = len(devices)
    is_verbose = len(params) == 1 or (n_devices == 1 and input_config["n_parallel"] == 1)

    arguments = {
        "split": input_config["split"],
        "dataset": input_config["dataset"],
        "n_workers": input_config["n_workers"],
        "algorithm": input_config["algorithm"],
        "devices": input_config["devices"],
        "is_verbose": is_verbose
    }

    print("Running", len(params), "training jobs")
    Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
        delayed(gather)(**arguments, **p) for p in tqdm(params, desc="Gathering experiment data",
                                                        position=0, leave=True))
