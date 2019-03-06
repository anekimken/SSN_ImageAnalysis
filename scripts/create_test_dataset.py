#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:17:57 2019

@author: adam

Creates and analyzes a 'trial' where the results are known
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Create image background using gaussian-distributed noise
mu, sigma = 100, 2.2  # mean and standard deviation
background = np.random.normal(mu, sigma, (65, 1200, 600))

# Create 3d gaussian kernel to represent mitochondria
xy_rad = 15
z_rad = 21
x_values = np.linspace(-xy_rad, xy_rad, xy_rad * 2 + 1)
y_values = np.linspace(-xy_rad, xy_rad, xy_rad * 2 + 1)
z_values = np.linspace(-z_rad, z_rad, z_rad * 2 + 1)
x, y, z = np.mgrid[-xy_rad:xy_rad + 1:1, -xy_rad:xy_rad + 1:1,
                   -z_rad:z_rad+1:1]
pos = np.empty(x.shape + (3,))
pos[:, :, :, 0] = x
pos[:, :, :, 1] = y
pos[:, :, :, 2] = z

rv = multivariate_normal.pdf(pos, mean=[0, 0, 0],
                             cov=[[xy_rad * 2, 0, 0],
                                  [0, xy_rad, 0],
                                  [0, 0, z_rad]])
multiplier = 200 / rv.max()
mito = multiplier * np.transpose(rv, (2, 0, 1))  # microscope convention

# Show plot of mitochondria from a couple angles just to be sure
max_proj = np.amax(mito, 2)  # collapse y axis
max_proj = max_proj.squeeze()
xz_fig, xz_ax = plt.subplots(figsize=(z_rad / 2, xy_rad / 2))
xz_ax.contourf(x_values, z_values, max_proj)
xz_ax.set_xlabel('z coord on microscope')
xz_ax.set_ylabel('x coord on microscope')

max_proj = np.amax(mito, 0)  # collapse z axis
max_proj = max_proj.squeeze()
xy_fig, xy_ax = plt.subplots(figsize=(4, 4))
xy_ax.contourf(x_values, y_values, max_proj)
xy_ax.set_xlabel('x coord on microscope')
xy_ax.set_ylabel('y coord on microscope')


def add_mito(image, coords, fake_mito):
    x_start = coords[1] - int(fake_mito.shape[1] / 2)
    x_end = coords[1] + int(fake_mito.shape[1] / 2 + 1)
    y_start = coords[2] - int(fake_mito.shape[2] / 2)
    y_end = coords[2] + int(fake_mito.shape[2] / 2 + 1)
    z_start = coords[0] - int(fake_mito.shape[0] / 2)
    z_end = coords[0] + int(fake_mito.shape[0] / 2 + 1)

    image[z_start:z_end, x_start:x_end, y_start:y_end] = fake_mito[:, :, :]

    return image


# Add mitochondria to the image
image = background.copy()
mito1_coords = [32, 500, 350]
mito2_coords = [30, 259, 320]
image = add_mito(image, mito1_coords, mito)
image = add_mito(image, mito2_coords, mito)

max_proj = np.amax(image, 0)  # collapse z axis
max_proj = max_proj.squeeze()
im_fig, im_ax = plt.subplots(figsize=(8, 16))
im_ax.imshow(max_proj, interpolation='none')
im_ax.set_xlabel('x coord on microscope')
im_ax.set_ylabel('y coord on microscope')

# Run analysis

# Save to results file format
# metadata.yaml
# trackpyBatchParams.yaml
# trackpyBatchParamsHistory.yaml
# trackpyBatchResults.yaml
# trackpyParamTestHistory.yaml
# unlinkedTrackpyBatchResults.yaml
