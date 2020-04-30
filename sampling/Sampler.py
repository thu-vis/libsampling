import numpy as np
import os
import csv
import sys
from sampling.SamplingMethods import *


class Sampler(object):
    def __init__(self):
        self.data = None
        self.category = None
        self.sampling_method = None

    def set_data(self, data, category):
        self.data = data
        self.category = category

    def set_sampling_method(self, method, **kwargs):
        self.sampling_method = method(**kwargs)

    def get_samples_idx(self):
        np.random.seed(0)
        if self.data is None:
            raise ValueError("Sampler.py: data not exist")
        if self.sampling_method is None:
            raise ValueError("Sampler.py: sampling method not specified")
        selected_indexes = self.sampling_method.sample(self.data, self.category)
        return selected_indexes

    def get_samples(self):
        selected_indexes = self.get_samples_idx()
        return self.data[selected_indexes], self.category[selected_indexes]
