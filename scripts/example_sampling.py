import numpy as np
from sampling.Sampler import *
from sampling.SamplingMethods import *
from matplotlib import pyplot as plt

print("run Sampler.py")
points = np.random.random((100, 2))
categories = np.random.random_integers(0, 1, 100)
sampling_method = RandomSampling

sampler = Sampler()

sampler.set_data(points, categories)
args = {
    'sampling_rate': 0.5
}
sampler.set_sampling_method(sampling_method, **args)

sampled_point, sampled_category = sampler.get_samples()

print(sampled_point, sampled_category)