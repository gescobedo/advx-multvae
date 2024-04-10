import numpy as np
from torch.utils.data import Dataset

from src.data.BaseDataset import BaseDataset
from src.data.FairDataset import FairDataset
from src.utils.helper import chunkify, chunkify_multi


class DeepMFDataset(Dataset):
    def __init__(self, dataset: BaseDataset, n_samples_per_iter=1, non_interaction_multiplier=5, seed=42):
        self.dataset = dataset
        self.n_samples_per_iter = n_samples_per_iter

        self.seed = seed
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.n_interactions = self.dataset.n_interactions
        self.non_interaction_multiplier = non_interaction_multiplier
        self.n_non_interactions = self.n_interactions * non_interaction_multiplier

        self.is_train_set = self.dataset.is_train_set
        if self.is_train_set:
            self.input_user_indices, self.input_item_indices = self.sample_non_interactions(self.dataset.data,
                                                                                            non_interaction_multiplier,
                                                                                            seed)

    @staticmethod
    def sample_non_interactions(data, non_interaction_multiplier, seed):
        n_users, n_items = data.shape
        n_interactions = int(data.sum().item())
        n_non_interactions = n_interactions * non_interaction_multiplier

        rng = np.random.default_rng(seed=seed)

        # 1 times more than intended to ensure that after filtering the interacted items,
        # we still have enough samples left
        s = rng.integers(low=0, high=n_users * n_items,
                         size=n_interactions * (non_interaction_multiplier + 1))
        sampled_item_indices = s // n_users
        sampled_user_indices = s % n_users

        sampled_data = data[sampled_user_indices, sampled_item_indices]

        sampled_indices = np.argwhere(sampled_data == 0)[:, 1]
        sampled_indices = rng.choice(sampled_indices,
                                     size=min(n_non_interactions, len(sampled_indices)),
                                     replace=False)

        interacted_user_indices, interacted_item_indices = data.nonzero()
        return np.concatenate([sampled_user_indices[sampled_indices], interacted_user_indices]), \
               np.concatenate([sampled_item_indices[sampled_indices], interacted_item_indices])

    def __len__(self):
        if self.is_train_set:
            return int(len(self.input_user_indices) // self.n_samples_per_iter)
        else:
            return self.n_users

    def __getitem__(self, idx):
        """
        Iterator mainly used by PyTorch DataLoader to train models,
        for validation and training check out 'iter_items()' and 'iter_users()'
        to use resources efficiently
        """
        indices = np.random.default_rng(seed=idx).integers(0, len(self.input_user_indices),
                                                           size=self.n_samples_per_iter)
        user_data = self.dataset.data[self.input_user_indices[indices], :]
        item_data = self.dataset.data[:, self.input_item_indices[indices]].T
        targets = self.dataset.data[self.input_user_indices[indices], self.input_item_indices[indices]]
        targets = np.asarray(targets).flatten()
        return indices, user_data, item_data, targets

    def iter_items(self):
        return chunkify(self.dataset.data.T, self.n_samples_per_iter)

    def iter_users(self):
        return chunkify_multi([self.dataset.data, self.dataset.targets], self.n_samples_per_iter)


class FairDeepMFDataset(DeepMFDataset):
    def __init__(self, dataset: FairDataset, n_samples_per_iter=1, non_interaction_multiplier=5, seed=42):
        super().__init__(dataset, n_samples_per_iter, non_interaction_multiplier, seed)
        self.user_groups_per_trait = dataset.user_groups_per_trait

    def __getitem__(self, idx):
        indices, *data = super().__getitem__(idx)
        traits = self.dataset.user_traits_encoding[self.input_user_indices[indices]].flatten()
        return indices, *data, traits

    def iter_users(self):
        return chunkify_multi([self.dataset.data, self.dataset.targets, self.dataset.user_traits_encoding],
                              self.n_samples_per_iter)
