#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:17:57 2019

@author: adam

Creates an image where the locations of "mitochondria" are known
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def create_test_stack(mito_coords, size,  show_mito=False):
    # Create image background using gaussian-distributed noise
    mu, sigma = 100, 2.2  # mean and standard deviation
    image = np.random.normal(mu, sigma, size)

    # Create 3d gaussian kernel to represent mitochondria
    xy_rad = 9
    z_rad = 13
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
                                 cov=[[xy_rad, 0, 0],
                                      [0, xy_rad, 0],
                                      [0, 0, z_rad]])
    multiplier = 200 / rv.max()
    mito = multiplier * np.transpose(rv, (2, 0, 1))  # microscope convention

    # Show plot of mitochondria from a couple angles just to be sure
    if show_mito is True:
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
        frame = coords['frame']
        x_start = coords['x'] - int(fake_mito.shape[2] / 2)
        x_end = coords['x'] + int(fake_mito.shape[2] / 2 + 1)
        y_start = coords['y'] - int(fake_mito.shape[1] / 2)
        y_end = coords['y'] + int(fake_mito.shape[1] / 2 + 1)
        z_start = coords['z'] - int(fake_mito.shape[0] / 2)
        z_end = coords['z'] + int(fake_mito.shape[0] / 2 + 1)

        image[frame, z_start:z_end,
              y_start:y_end, x_start:x_end] = fake_mito[:, :, :]

        return image

    # Add mitochondria to the image
    for i in range(len(mito_coords)):
        image = add_mito(image,
                         mito_coords.iloc[i][['z', 'y', 'x', 'frame']],
                         mito)

    return image


if __name__ == '__main__':
    mito_coords = pd.DataFrame([[30, 200, 350, 0],
                                [30, 250, 350, 0],
                                [30, 300, 350, 0],
                                [30, 350, 350, 0]],
                               columns=['z', 'y', 'x', 'frame'])
    image = create_test_stack(mito_coords, (1, 61, 1200, 600))
    max_proj = np.amax(image[0], 0)  # collapse z axis
    max_proj = max_proj.squeeze()
    im_fig, im_ax = plt.subplots(figsize=(8, 16))
    im_ax.imshow(max_proj, interpolation='none')
    im_ax.set_xlabel('x coord on microscope')
    im_ax.set_ylabel('y coord on microscope')
