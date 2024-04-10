import os
import json
import numpy as np
import pandas as pd
from scipy import sparse as sp


def store_results(storage_dir, interaction_matrix, user_info, attribute_descriptions, item_info=None,
                  user_mapping=None, item_mapping=None):
    os.makedirs(storage_dir, exist_ok=True)
    sp.save_npz(os.path.join(storage_dir, "interactions.npz"), interaction_matrix)

    user_info.to_csv(os.path.join(storage_dir, "user_info.csv"), index=False)

    if item_info is not None:
        item_info.to_csv(os.path.join(storage_dir, "item_info.csv"), index=False)

    if user_mapping is not None:
        user_mapping.to_csv(os.path.join(storage_dir, "user_mapping.csv"), index=False)

    if item_mapping is not None:
        item_mapping.to_csv(os.path.join(storage_dir, "item_mapping.csv"), index=False)

    with open(os.path.join(storage_dir, "attribute_descriptions.json"), "w") as fh:
        json.dump(attribute_descriptions, fh, indent="\t")


def ensure_min_interactions(interaction_matrix, min_interactions_user, min_interactions_item,
                            user_info=None, item_info=None):
    n_users, n_items = interaction_matrix.shape
    item_mapping = pd.DataFrame.from_dict(
        {"old": item_info.itemID if item_info is not None else range(n_items), "new": range(n_items)})
    user_mapping = pd.DataFrame.from_dict(
        {"old": user_info.userID if user_info is not None else range(n_users), "new": range(n_users)})

    # Remove until there are enough interactions from each side
    while True:
        n_interactions_per_user = np.array(interaction_matrix.sum(axis=1)).flatten()
        n_interactions_per_item = np.array(interaction_matrix.sum(axis=0)).flatten()

        # filter items with too less interactions
        enough_interactions_item = n_interactions_per_item >= min_interactions_item
        interaction_matrix = interaction_matrix[:, enough_interactions_item]

        # only keep those users with enough interactions
        enough_interactions_user = n_interactions_per_user >= min_interactions_user
        interaction_matrix = interaction_matrix[enough_interactions_user]

        item_mapping = item_mapping[enough_interactions_item].reset_index(drop=True)
        item_mapping = item_mapping.assign(new=item_mapping.index)

        user_mapping = user_mapping[enough_interactions_user].reset_index(drop=True)
        user_mapping = user_mapping.assign(new=user_mapping.index)

        if np.sum(enough_interactions_item == False) == 0 \
                and np.sum(enough_interactions_user == False) == 0:
            break

    print("Final shape of interactions matrix is", interaction_matrix.shape)
    print("==> {} users and {} items are remaining.\n".format(*interaction_matrix.shape))
    returns = [interaction_matrix, user_mapping, item_mapping, None, None]

    if user_info is not None:
        user_info = user_info[user_info["userID"].isin(set(user_mapping["old"]))]
        udict = {old: new for old, new in zip(user_mapping["old"], user_mapping["new"])}
        user_info = user_info.assign(userID=user_info["userID"].replace(udict))
        user_info.reset_index(drop=True, inplace=True)
        returns[-2] = user_info

    if item_info is not None:
        item_info = item_info[item_info["itemID"].isin(set(item_mapping["old"]))]
        idict = {old: new for old, new in zip(item_mapping["old"], item_mapping["new"])}
        item_info = item_info.assign(itemID=item_info["itemID"].replace(idict))
        item_info.reset_index(drop=True, inplace=True)
        returns[-1] = item_info

    return returns


def print_stats(interaction_matrix):
    n_users = interaction_matrix.shape[0]
    n_items = interaction_matrix.shape[1]
    n_interactions = int(interaction_matrix.sum())
    density = n_interactions / (n_items * n_users)

    print(f"Number of interactions is {n_interactions},")
    print(f"which leads to a density of {density:.4f}.")
