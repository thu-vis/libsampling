import os
import numpy as np
from sampling.Sampler import *
from sampling.SamplingMethods import *

print("run Sampler.py")

# Simple synthetic data
points = np.random.random((10000, 2)) # Generated data, the input data should be a numpy array with the shape (n, 2)
categories = np.random.randint(0, 10, 10000) # Generated label, multi-class sampling method would consider the label information as an reason to select or not select an item. It would be a np.zeros(n) as default.

# Datasets used in our study
all_data = np.load(os.path.join('data', 'abalone.npz'))
points, categories = all_data['positions'], all_data['labels']

print(points.shape, categories.shape)

sampler = Sampler()

sampler.set_data(points, categories) # For single-class sampling methods like random sampling, categories is not needed to be provided
sampling_method = RandomSampling # You can choose your desired sampling method.
rs_args = {
    'sampling_rate': 0.3 # You can set the sampling ratio and other specific params for different sampling methods here.
}

sampler.set_sampling_method(sampling_method, **rs_args) # Set Random Sampling for the sampler with necessary params
sampled_point, sampled_category = sampler.get_samples() # Get the sampling result

print("Random sampling result:")
print(sampled_point, sampled_category)

sampling_method = OutlierBiasedRandomSampling
outlier_score = np.sum(np.abs(points - 0.5), axis=1)
obrs_args = {
    'sampling_rate': 0.5, # You can set the specific params for different sampling methods here, e.g., sampling rate
    'outlier_score': outlier_score # The default outlier_score will be determined by the class purity if you do not pass your own outlier_score to outlier biased sampling methods
}

sampler.set_sampling_method(sampling_method, **obrs_args) # Set Outlier Biased Random Sampling for the sampler with necessary params
sampled_point, sampled_category = sampler.get_samples() # Get the sampling result

print("Outlier biased random sampling result:")
print(sampled_point, sampled_category)

sampling_method = RecursiveSubdivisionBasedSampling
rsbs_args = { # This sampling method do not need sampling rate as input
	'canvas_width': 1600,
	'canvas_height': 900,
	'grid_width': 20,
	'threshold': 0.02,
	'occupied_space_ratio': 0.02,
	'backtracking_depth': 4
}
sampler.set_sampling_method(sampling_method, **rsbs_args)
sampled_point, sampled_category = sampler.get_samples() # Get the sampling result

print("Recursive subdivision based sampling result:")
print(sampled_point, sampled_category)