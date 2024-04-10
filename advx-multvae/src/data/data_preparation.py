import os
import numpy as np
import pandas as pd
from enum import Enum
from scipy import sparse as sp
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold, train_test_split
from src.utils.helper import yaml_dump, json_load, json_dump


class ResamplingStrategy(str, Enum):  # inherit from str for serialization, see https://stackoverflow.com/a/51976841
    NoResampling = "none"
    Undersample = "undersample"
    Oversample = "oversample"

    def __str__(self):
        return self.value


def resample_dataset(strategy: ResamplingStrategy, interaction_matrix, user_info: pd.DataFrame,
                     features: list, random_state: int, strategy_params: dict = None):
    if len(features) != 1 and strategy != ResamplingStrategy.NoResampling:
        raise AttributeError("For experiments with multiple features, dataset resampling is not available.")

    # ensure params variable is not None
    strategy_params = strategy_params or {}

    if strategy == ResamplingStrategy.NoResampling:
        return interaction_matrix, user_info[features]
    else:
        feature = features[0]
        sampler_cls = RandomUnderSampler if strategy == ResamplingStrategy.Undersample else RandomOverSampler
        sampler = sampler_cls(**strategy_params, random_state=random_state)

        interaction_matrix_sampled, user_info_sampled = sampler.fit_resample(X=interaction_matrix, y=user_info[feature])

        # generate new dataframe for oversampled data
        user_info_sampled = pd.DataFrame(user_info_sampled, columns=[feature])
        user_info_sampled["userID"] = user_info_sampled.index
        user_info_sampled = user_info_sampled[["userID", features]]  # switch order of columns
        return interaction_matrix_sampled, user_info_sampled


def split_interactions(interaction_matrix, test_size=0.8, random_state=42):
    user_idxs, item_idxs, _ = sp.find(interaction_matrix == 1)

    tr_ind, te_ind = train_test_split(np.arange(len(user_idxs)), test_size=test_size, random_state=random_state)

    tr_values = np.ones(len(tr_ind))
    tr_matrix = sp.csr_matrix((tr_values, (user_idxs[tr_ind], item_idxs[tr_ind])), shape=interaction_matrix.shape)

    te_values = np.ones(len(te_ind))
    te_matrix = sp.csr_matrix((te_values, (user_idxs[te_ind], item_idxs[te_ind])), shape=interaction_matrix.shape)

    return tr_matrix, te_matrix


def drop_users_no_inter(interaction_matrix_train, interaction_matrix_test, df_user_info):
    """
    For datasets where users have only very few interactions with items, it may happen that
    a user has no interactions in a specific split. We mitigate this by filtering out such users.
    """
    zero_mask = (interaction_matrix_train.sum(axis=1) == 0) | (interaction_matrix_test.sum(axis=1) == 0)
    zero_mask = np.array(zero_mask).flatten()
    interaction_matrix_train = interaction_matrix_train[~zero_mask, :]
    interaction_matrix_test = interaction_matrix_test[~zero_mask, :]

    df_user_info = df_user_info.loc[~zero_mask, :]
    df_user_info.reset_index(drop=True, inplace=True)
    df_user_info = df_user_info.assign(userID=df_user_info.index)
    return interaction_matrix_train, interaction_matrix_test, df_user_info


def select_and_reset(df_user_info, indices):
    df_user_info = df_user_info.iloc[indices].copy()
    df_user_info.reset_index(drop=True, inplace=True)
    df_mapping = pd.DataFrame.from_dict({"old": df_user_info["userID"].copy(), "new": df_user_info.index})
    df_user_info["userID"] = df_user_info.index
    return df_user_info, df_mapping


def split_and_store(split_indices, interaction_matrix, df_user_info, random_state,
                    storage_dir, split_abbrev):
    """
    Performs the data preparation for validation or test data. Results will then be stores.
    """
    im = interaction_matrix[split_indices, :]
    tr_im, te_im = split_interactions(im, random_state=random_state)

    df_user_info, df_mapping = select_and_reset(df_user_info, split_indices)
    tr_im, te_im, df_user_info = drop_users_no_inter(tr_im, te_im, df_user_info)

    df_user_info.to_csv(os.path.join(storage_dir, f"{split_abbrev}_user_info.csv"), index=False)
    df_mapping.to_csv(os.path.join(storage_dir, f"{split_abbrev}_user_mapping.csv"), index=False)

    sp.save_npz(os.path.join(storage_dir, f"{split_abbrev}_input.npz"), tr_im)
    sp.save_npz(os.path.join(storage_dir, f"{split_abbrev}_target.npz"), te_im)


def perform_kfold_split(n_folds: int,
                        interaction_matrix,
                        df_user_info: pd.DataFrame,
                        storage_dir: str, features: list,
                        resampling_strategy: ResamplingStrategy,
                        random_state: int = 42):
    print(f"Creating {n_folds} folds for cross validation")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # generate splits
    X = np.arange(interaction_matrix.shape[0])
    fold_indices = [indices for _, indices in kf.split(X)]

    # Split data into different folds and store them to reduce compute time while training
    for i in range(n_folds):
        print("Splitting data for fold", i)
        fold_interaction_matrix = interaction_matrix.copy()

        fold_dir = os.path.join(storage_dir, str(i))
        os.makedirs(fold_dir, exist_ok=True)

        n_tr_folds = n_folds - 2
        tr_indices = np.concatenate([fold_indices[i] for i in (i + np.arange(n_tr_folds)) % n_folds])
        tr_indices = np.sort(tr_indices)

        tr_im = fold_interaction_matrix[tr_indices, :]
        tr_user_info, df_mapping = select_and_reset(df_user_info, tr_indices)

        # Ensure that all items in the training data feature user interactions
        zero_mask = np.array(tr_im.sum(axis=0) == 0).flatten()
        dropped_item_indices = np.argwhere(zero_mask).flatten().tolist()
        yaml_dump({"dropped_item_indices": dropped_item_indices}, os.path.join(fold_dir, "dropped_items.yaml"))

        tr_im = tr_im[:, ~zero_mask]
        fold_interaction_matrix = fold_interaction_matrix[:, ~zero_mask]

        tr_im_os, df_y_os = resample_dataset(strategy=resampling_strategy, interaction_matrix=tr_im,
                                             user_info=tr_user_info, features=features, random_state=random_state)

        df_y_os.to_csv(os.path.join(fold_dir, "train_user_info.csv"), index=False)
        sp.save_npz(os.path.join(fold_dir, "train_input.npz"), tr_im_os)

        # ===== Validation data ======
        vd_indices = np.sort(fold_indices[(n_tr_folds + i) % n_folds])
        split_and_store(vd_indices, fold_interaction_matrix, df_user_info, random_state, fold_dir, split_abbrev="val")

        # ===== Test data ======
        te_indices = np.sort(fold_indices[(n_tr_folds + i + 1) % n_folds])
        split_and_store(te_indices, fold_interaction_matrix, df_user_info, random_state, fold_dir, split_abbrev="test")

        # ===== Validating the results =====
        # Ensure that no indices overlap between the different data sets
        n_indices_total = len(tr_indices) + len(vd_indices) + len(te_indices)
        all_indices = np.union1d(np.union1d(tr_indices, vd_indices), te_indices)
        assert n_indices_total == len(all_indices), f"User indices of different splits overlap in fold {i}"


def ensure_make_data(data_dir: str, n_folds: int, target_path: str, features: list = None,
                     resampling_strategy: ResamplingStrategy = ResamplingStrategy.NoResampling, random_state: int = 42):
    """
    Ensure that dataset is prepared for experiments by performing `n_folds` fold splitting for cross-validation and
    possible resampling.

    @data_dir: the path to the dataset
    @n_folds: the number of folds for k-fold splitting
    @target_path: where the resulting folds should be stored
    @features: list of user features that will be kept in the dataset, defaults to None to keep all features
    @resampling_strategy: whether and how to perform resampling of the dataset
    @random_state: the random state for the experiments
    """
    state_file = os.path.join(target_path, "used_config.json")
    prev_state = None
    if os.path.exists(state_file):
        try:
            prev_state = json_load(state_file)
        except:
            pass

    current_state = {
        "random_state": random_state,
        "resampling_strategy": resampling_strategy,
        "features": features
    }

    states_match = prev_state is not None \
                   and prev_state["random_state"] == current_state["random_state"] \
                   and prev_state["resampling_strategy"] == current_state["resampling_strategy"] \
                   and len(set(current_state["features"] or []) - set(prev_state["features"])) == 0

    if not states_match:
        interaction_matrix = sp.load_npz(os.path.join(data_dir, "interactions.npz"))
        df_user_info = pd.read_csv(os.path.join(data_dir, "user_info.csv"))

        features = features or list(df_user_info.columns)
        features_not_found = set(features) - set(df_user_info.columns)
        if len(features_not_found) > 0:
            raise AttributeError(f"Dataset does not contain the user features {features_not_found}")
        current_state["features"] = features

        print(f"Preparing dataset for experiments on features {features}")
        perform_kfold_split(n_folds, interaction_matrix, df_user_info,
                            storage_dir=target_path,
                            features=features,
                            random_state=random_state,
                            resampling_strategy=resampling_strategy
                            )

        # store state so that we don't repeat processing the features
        json_dump(current_state, state_file)
