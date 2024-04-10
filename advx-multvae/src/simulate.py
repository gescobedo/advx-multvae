import os
import shutil
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from rmet import calculate
from tqdm import tqdm
from joblib import Parallel, delayed

import torch

import global_config
from src.data.DynamicFeedbackDataset import DynamicFeedbackDataset
from src.logging.utils import redirect_to_tqdm
from src.algorithms.utils import validate_epoch
from src.utils.adv_evaluation import eval_adversaries
from src.utils.input_validation import parse_input
from src.algorithms.vae import setup_model_and_optimizers_vae, preprocess_data_vae
from src.data.data_loading import get_datasets_and_loaders, get_datasets_and_loaders_mf, sparse_tensor_to_sparse_scipy
from src.algorithms.mf import setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf
from src.utils.training_utils import check_run_dict, run_to_fold_dict, adjust_results_dir
from src.utils.helper import reproducible, yaml_load, json_dump, \
    load_model_from_path, get_device_matching_current_process, pickle_dump, yaml_dump

algorithm_fn_mapping = {
    "mf": (get_datasets_and_loaders_mf, setup_model_and_optimizers_mf, preprocess_data_mf, validate_epoch_mf),
    "vae": (get_datasets_and_loaders, setup_model_and_optimizers_vae, preprocess_data_vae, validate_epoch),
}


def simulate(algorithm, dataset, split, fold, model_path, experiment_config, simulation_config,
             log_dir, n_workers, devices, is_verbose=False):
    reproducible(global_config.EXP_SEED)
    rng = np.random.default_rng(global_config.EXP_SEED)

    os.makedirs(log_dir, exist_ok=True)

    # Store used config for later retrieval
    yaml_dump(simulation_config, os.path.join(log_dir, "simulation_config.yaml"))

    device = get_device_matching_current_process(devices)
    loading_fn, setup_fn, preprocess_fn, validation_fn = algorithm_fn_mapping[algorithm]

    # Probabilities of user selecting one of the top k items, based on nDCG's discount factor
    discount_factors = 1 / np.log2(np.arange(1, simulation_config["top_k_display_for_sampling"] + 1) + 1)
    sampling_probability = discount_factors / discount_factors.sum()

    with redirect_to_tqdm():
        # Load data
        dataset_and_loaders = loading_fn(dataset_name=dataset, fold=fold,
                                         splits=("train", split), run_parallel=True, n_workers=n_workers,
                                         **experiment_config["data_loader"]
                                         )
        tr_set, tr_loader = dataset_and_loaders["train"]
        eval_set, eval_loader = dataset_and_loaders[split]
        dyn_eval_set = DynamicFeedbackDataset(eval_set)

        model, *_, loss_fn, loss_fn_adv = setup_fn(experiment_config, tr_set.n_users, tr_set.n_items, device)
        load_model_from_path(model, model_path)
        model.eval()

        perform_adv_training = experiment_config["adv"]["perform_adv_training"]

        if is_verbose:
            print(f"Simulation on '{split}' split for model '{model_path}'")

        with torch.no_grad():
            initial_targets = []
            for sim in range(simulation_config["n_simulation_rounds"]):
                gathered_data = defaultdict(lambda: [])
                added_feedbacks = []

                bacc = 0
                sample_count = 0
                dataset_metrics = defaultdict(lambda: defaultdict(lambda: 0))
                user_metrics = defaultdict(lambda: defaultdict(lambda: []))
                for indices, *model_input, targets, adv_targets in tqdm(eval_loader,
                                                                        desc=f"Simulation [{sim}|{simulation_config['n_simulation_rounds']}]",
                                                                        position=1, leave=True, disable=not is_verbose):
                    n_samples = len(indices)
                    sample_count += n_samples

                    if sim == 0:
                        initial_targets.append(targets)

                    targets = targets.to(device)
                    adv_targets = adv_targets.flatten()
                    adv_targets = adv_targets.to(device, dtype=torch.long)
                    model_input = preprocess_fn(model_input, device)
                    logits, loss_dict, adv_logits = model(*model_input)

                    if perform_adv_training:
                        _, adv_bacc = eval_adversaries(adv_logits, adv_targets, loss_fn_adv)
                        bacc += adv_bacc.cpu().item() * n_samples

                    # Removing items from input data
                    logits[model_input[0].nonzero(as_tuple=True)] = .0

                    top_k_display_for_sampling = simulation_config["top_k_display_for_sampling"]
                    top_k_recommendations_to_store = simulation_config["top_k_recommendations_to_store"]
                    top_k_indices = torch.topk(logits, max(top_k_display_for_sampling, top_k_recommendations_to_store),
                                               dim=-1).indices.cpu()

                    # Gather metrics, aggregated over the whole user base
                    metric_results = calculate(model_config["metrics"], logits, targets, model_config["top_k"],
                                               return_aggregated=True, return_individual=True)
                    for k, v in metric_results.items():
                        for t, v1 in v.items():
                            if "_individual" in k:
                                user_metrics[k[:-len("_individual")]][t].append(v1)
                            else:
                                dataset_metrics[k][t] += v1 * n_samples

                    # Note that, for simplicity, for simulating interactions we ignore the fact that commonly,
                    # male users on average interact more often with the system then female users
                    n_items_to_sample = rng.integers(low=simulation_config["user_sample_n_items_min"],
                                                     high=simulation_config["user_sample_n_items_max"] + 1,
                                                     size=n_samples)

                    feedback = {}
                    for uid, nits, tki in zip(indices, n_items_to_sample,
                                              top_k_indices[:, :top_k_display_for_sampling]):
                        feedback[uid.item()] = rng.choice(tki.numpy(), size=nits, replace=False, p=sampling_probability)
                    added_feedback = dyn_eval_set.include_feedback(feedback)
                    gathered_data["user_indices"].append(indices)
                    gathered_data["recommendations"].append(top_k_indices[:, :top_k_recommendations_to_store])
                    gathered_data["adv_targets"].append(adv_targets.cpu())

                    # aggregate feedbacks while ensure that order of users is maintained
                    added_feedbacks.extend([added_feedback[i.item()] for i in indices])

                # apply feedback to the dataset for the next simulation step
                dyn_eval_set.apply_feedback()

                if sim % simulation_config.get("store_simulation_results_every", 1) == 0:
                    # store individual users metrics
                    user_metrics = {
                        k: {t: torch.concat(v1).cpu().numpy() for t, v1 in v.items() if isinstance(v1[0], torch.Tensor)}
                        for k, v in user_metrics.items()}
                    pickle_dump(user_metrics, os.path.join(log_dir, f"sim_{sim}_user_metrics.pkl"))

                    # store aggregated metrics over the whole evaluation dataset, i.e., all users
                    dataset_metrics = {k: {t: v1 / sample_count for t, v1 in v.items()} for k, v in
                                       dataset_metrics.items()}
                    dataset_metrics["adv_bacc"] = bacc / sample_count
                    yaml_dump(dataset_metrics, os.path.join(log_dir, f"sim_{sim}_dataset_metrics.yaml"))

                    # store all other data
                    gathered_data = {k: torch.concat(v).cpu().numpy() for k, v in gathered_data.items()}
                    pickle_dump(gathered_data, os.path.join(log_dir, f"sim_{sim}_gathered_data.pkl"))

                # need to store feedback each step, otherwise we won't know the items in the input data
                pickle_dump({"added_feedback": added_feedbacks}, os.path.join(log_dir, f"sim_{sim}_added_feedback.pkl"))

            # store initial targets to later on only store the changes, thus saving disk space
            initial_targets = sparse_tensor_to_sparse_scipy(torch.concat(initial_targets).cpu().to_sparse())
            sp.save_npz(os.path.join(log_dir, "initial_targets.npz"), initial_targets)

            # store user feature - value mapping
            feature_map = {k: v.value_map for k, v in eval_set.user_features.items()}
            yaml_dump(feature_map, os.path.join(log_dir, "user_feature_map.yaml"))


if __name__ == "__main__":

    input_config = parse_input("simulate", options=["run", "experiment", "gpus", "split", "config",
                                                    "n_folds", "model_pattern", "n_parallel", "n_workers"],
                               access_as_properties=False)

    run_dict = input_config["run_dict"]
    if not check_run_dict(run_dict):
        exit()
    fold_dict = run_to_fold_dict(run_dict)

    params = []
    for fold in range(input_config["n_folds"]):
        for run_dir, run_data in fold_dict[fold].items():
            result_dirs = adjust_results_dir(run_dir, lambda _: "sim")
            model_config = yaml_load(run_data["config_file"])
            simulation_config = yaml_load(input_config["config"])

            for model_name, model_path in run_data["models"]:
                result_dir = os.path.join(result_dirs, model_name)
                # clear data from previous run
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir, ignore_errors=True)

                params.append({
                    "fold": fold,
                    "log_dir": result_dir,
                    "model_path": model_path,
                    "experiment_config": model_config,
                    "simulation_config": simulation_config
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

    print("Running", len(params), "simulation jobs")
    Parallel(n_jobs=min(n_devices, len(params)), verbose=11)(
        delayed(simulate)(**arguments, **p) for p in tqdm(params, desc="Simulating user interactions",
                                                          position=0, leave=True))
