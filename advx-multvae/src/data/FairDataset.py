import os
import pandas as pd
from typing import List

import torch

from src.data.BaseDataset import BaseDataset
from src.data.user_feature import UserFeature, FeatureDefinition, FeatureType


class FairDataset(BaseDataset):
    """
    Base fairness dataset class that all fair datasets should build upon
    """

    def __init__(self, data_dir: str, features: List[FeatureDefinition], split: str = "train", transform=None):
        super().__init__(data_dir, split, transform)
        self.features = features
        self.feature_names = [feat.name for feat in features]

        user_info = pd.read_csv(os.path.join(data_dir, split + "_user_info.csv"))
        self.user_features = {feat.name: UserFeature(feat, user_info[feat.name]) for feat in features}

    def __getitem__(self, idx):
        x_sample, y_sample = super().__getitem__(idx)
        feature_values = [self.user_features[feature.name].get_values()[idx] for feature in self.features]
        return idx, x_sample, y_sample, feature_values  # TODO: generic dataloader?
