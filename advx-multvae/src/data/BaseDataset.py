import os
from scipy import sparse
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset class that all datasets should build upon
    """

    def __init__(self, data_dir, split='train', transform=None):
        super().__init__()

        self.which = split
        self.data_dir = data_dir
        self.transform = transform
        self.is_train_set = split == "train"

        # Determine input file and load data
        inputs_file_name = split + "_input.npz"
        self.data = sparse.load_npz(os.path.join(self.data_dir, inputs_file_name))

        # Determine target file and load data
        targets_file_name = split + "_target.npz"

        if self.is_train_set:
            # During training, we want to recreate the input
            self.targets = self.data
        else:
            self.targets = sparse.load_npz(os.path.join(self.data_dir, targets_file_name))

        self.n_users = self.data.shape[0]
        self.n_items = self.data.shape[1]
        self.n_interactions = self.data.sum()
        if not self.is_train_set:
            self.n_interactions += self.targets.sum()

        self.__ensure_types()

    def __len__(self):
        return self.n_users

    def __ensure_types(self):
        self.data = self.data.astype("float32")
        self.targets = self.targets.astype("float32")

    def __getitem__(self, idx):
        x_sample = self.data[idx, :].toarray().squeeze()
        if self.transform:
            x_sample = self.transform(x_sample)

        y_sample = self.targets[idx, :].toarray().squeeze()

        return x_sample, y_sample
