from ast import literal_eval
from typing import List, Union

import numpy as np
from enum import Enum
from dataclasses import dataclass


class FeatureType(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 3
    CONTINUOUS_SEQUENCE = 4

    @classmethod
    def from_str(cls, s):
        m = {
            "categorical": cls.CATEGORICAL,
            "continuous": cls.CONTINUOUS,
            "continuous_sequence": cls.CONTINUOUS_SEQUENCE,
        }
        if s in m:
            return m[s]
        else:
            raise AttributeError(f"'{s}' not a valid feature type. Choose from {list(m.keys())}")


@dataclass
class FeatureDefinition:
    name: str
    type: FeatureType


class UserFeature:
    def __init__(self, feature: FeatureDefinition, raw_values: List[Union[int, float, str]]):
        """
        Helper class for different user features to ease their handling

        :param feature: the definition of the feature
        :param raw_values: the raw feature values for the individual users, can either be numeric values (int/float)
                            or string representations of a list of values
        """

        self.feature = feature
        self.is_categorical_feature = self.feature.type == FeatureType.CATEGORICAL
        self.is_sequential_feature = self.feature.type == FeatureType.CONTINUOUS_SEQUENCE
        self.is_continuous_feature = self.feature.type == FeatureType.CONTINUOUS
        self.n_values = len(raw_values)

        self._raw_values = raw_values
        if self.is_sequential_feature:
            self.sequence_values = np.stack([literal_eval(val) for val in self._raw_values])

        if self.is_categorical_feature:
            self.unique_values = tuple(sorted(set(self._raw_values)))
            self.value_map = {lbl: i for i, lbl in enumerate(self.unique_values)}
            self.encoded_values = np.array([self.value_map[lbl] for lbl in self._raw_values], dtype=int)
            self.value_indices_groups = {lbl: np.argwhere(self.encoded_values == self.value_map[lbl]).flatten()
                                         for lbl in self.unique_values}

    def get_values(self):
       # if self.is_continuous_feature:
        #    self._raw_values = (self._raw_values-self._raw_values.min())/(self._raw_values.max()-self._raw_values.min())
        if self.is_categorical_feature:
            return self.encoded_values
        else:
            return self.sequence_values if self.is_sequential_feature else self._raw_values

    def count(self):
        if self.is_categorical_feature:
            return {k: len(v) for k, v in self.value_indices_groups.items()}
        else:
            return self.n_values

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"UserFeature(name={self.feature.name}, type={self.feature.type}, counts={self.count()})"
