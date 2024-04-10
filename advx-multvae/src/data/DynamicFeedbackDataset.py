from torch.utils.data import Dataset
from src.data.BaseDataset import BaseDataset


class DynamicFeedbackDataset(Dataset):
    def __init__(self, dataset):
        assert (isinstance(dataset, BaseDataset))

        # maintain reference to original dataset, as this allows us to include feedback,
        # which is immediately reflected on the dataloader
        # see https://discuss.pytorch.org/t/solved-will-change-in-dataset-be-reflected-on-dataloader-automatically/10206
        self.dataset = dataset
        self.temporary_data = None
        self.temporary_targets = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def include_feedback(self, feedback: dict):
        """
        Includes users feedback into the dataset
        feedback ... dictionary of {user_id: list of item_id mappings} to indicate with
                     which items the user has interacted

        For efficiency reasons, this call needs to be followed up by 'apply_feedback', to make the
        feedback available in the dataset. More concretely, we need slicing during data loading, which
        is more efficient in csr matrices, but changing / setting the values is more efficient for lil matrices.
        """
        if self.temporary_data is None:
            self.temporary_data = self.dataset.data.tolil()
            self.temporary_targets = self.dataset.targets.tolil()

        added_feedback = {}
        for uid, f in feedback.items():
            # Check which of the suggestions aren't already targets (input matched should already be removed)
            added_feedback[uid] = f[~((self.temporary_data[uid, f] != 0).toarray().flatten())]

            # Feedback may now be used as input ...
            self.temporary_data[uid, f] = 1

            # ... and not anymore as target
            self.temporary_targets[uid, f] = 0
        return added_feedback

    def apply_feedback(self):
        self.dataset.data = self.temporary_data.tocsr()
        self.dataset.targets = self.temporary_targets.tocsr()
        self.temporary_data = None
        self.temporary_targets = None
