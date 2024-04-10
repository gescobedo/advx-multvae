import os
from typing import Union, List, Tuple
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader
from joblib.externals.loky.backend import get_context

import global_config
from data_paths import get_data_path
from src.data.FairDataset import FairDataset
from src.data.DeepMFDataset import FairDeepMFDataset
from src.data.data_preparation import ensure_make_data
from src.data.user_feature import FeatureDefinition


def sparse_scipy_to_tensor(matrix):
    return torch.sparse_coo_tensor(*sparse_scipy_to_tensor_params(matrix))


def sparse_scipy_to_tensor_params(matrix):
    # sparse tensor multiprocessing in dataloaders is not supported,
    # therefore we will create the sparse tensor only in training loop
    m = matrix.tocoo()
    indices = torch.stack([torch.tensor(m.row), torch.tensor(m.col)])
    return indices, m.data, m.shape


def sparse_tensor_to_sparse_scipy(tensor: torch.Tensor):
    return sp.coo_matrix((tensor._values(), tensor._indices()), shape=tensor.shape)


def train_collate_fn(data):
    # data must not be batched (not supported by PyTorch layers)
    indices, user_data, item_data, targets = data
    user_data = sparse_scipy_to_tensor_params(user_data)
    item_data = sparse_scipy_to_tensor_params(item_data)
    targets = torch.tensor(targets)
    return indices, user_data, item_data, targets


def train_collate_fn_fair(data):
    *data, traits = data
    return *train_collate_fn(data), torch.tensor(traits)


def get_datasets_and_loaders(dataset_name: str, fold: int, splits: Tuple[str],
                             features: List[FeatureDefinition],
                             batch_size: Union[int, None] = 64, n_workers=0,
                             shuffle_train=True, run_parallel=False,
                             transform=None, random_state=global_config.EXP_SEED):
    data_path = get_data_path(dataset_name)
    ensure_make_data(data_path, n_folds=global_config.MAX_FOLDS, target_path=data_path, random_state=random_state,
                     features=[f.name for f in features])

    dataset_loader_dict = {}
    for split in splits:
        is_train_split = split == "train"
        dataset = FairDataset(data_dir=os.path.join(data_path, str(fold)),
                              split=split, features=features, transform=transform)

        # for multiprocessing (with joblib) we need to set a different multiprocessing context
        # https://github.com/pytorch/pytorch/issues/44687
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            shuffle=is_train_split and shuffle_train, pin_memory=True,
                            multiprocessing_context=get_context("loky") if run_parallel and n_workers > 0 else None)
        dataset_loader_dict[split] = (dataset, loader)

    return dataset_loader_dict


def get_datasets_and_loaders_mf(dataset_name, features: List[FeatureDefinition], fold, splits=("train",),
                                n_workers=global_config.N_WORKERS,
                                shuffle_train=True, run_parallel=False,
                                transform=None, random_state=global_config.EXP_SEED, n_samples_per_iter=64,
                                non_interaction_multiplier=7):
    dataset_loader_dict = get_datasets_and_loaders(dataset_name, fold, splits, features, None, n_workers,
                                                   shuffle_train, run_parallel, transform, random_state)

    for split, (dataset, loader) in dataset_loader_dict.items():
        mf_dataset = FairDeepMFDataset(dataset, n_samples_per_iter, non_interaction_multiplier, random_state)
        # - Based on PyTorch source code it doesn't seem possible to just modify the dataset the dataloader uses,
        # as we'd have to adjust the internal RandomSampler instance.
        # - Moreover, sparse batched tensors do not work with multiprocessing and linear layers yet.
        # - We cannot use pin_memory (which often helps speeding up data loading) as we have sparse tensors
        mp_context = get_context("loky") if run_parallel and n_workers > 0 else None
        mf_loader = DataLoader(mf_dataset, batch_size=None, num_workers=n_workers,
                               shuffle=dataset.is_train_set and shuffle_train, pin_memory=True,
                               multiprocessing_context=mp_context, collate_fn=train_collate_fn_fair)
        dataset_loader_dict[split] = (mf_dataset, mf_loader)
    return dataset_loader_dict
