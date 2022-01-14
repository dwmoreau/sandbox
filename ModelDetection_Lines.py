"""
Correlate the inverse of absorption with the inverse of the image
Try fitting two lines to the curves across the 
"""
import sys
sys.path.append('/home/david/dials_dev/modules/dxtbx/src')
sys.path.append('/home/david/dials_dev/build/lib')
sys.path.append('/home/david/dials_dev/modules')

from cctbx import factor_kev_angstrom
from dials.array_family import flex
import dxtbx
#from dxtbx.model.experiment_list import ExperimentListFactory
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyFAI.geometry
import pyFAI.azimuthalIntegrator
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from skimage import feature
from skimage.filters import median
from skimage.filters import sobel
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity

import Modules as MOD


def FindMaxAbs(normalized_image):
    rows = 100
    nrows = np.int(normalized_image.shape[0] / rows)
    sides = 1
    xy = []
    xy_min = []
    for index in range(nrows):
        #fig, axes = plt.subplots(1, 1)
        curve = np.nanmean(normalized_image[rows * index: rows * (index+1), sides: -sides], axis=0)
        #axes.plot(curve)
        filtered_inverse = gaussian_filter1d(1/curve, 1)
        peaks = find_peaks(filtered_inverse, width=30, prominence=0.05)[0]
        #axes.plot(peaks, curve[peaks], marker='x', linestyle='none')
        for peak in peaks:
            xy.append([
                sides + peak, (rows * index + rows * (index+1)) / 2
                ])
        #plt.show()
    xy = np.array(xy)
    tolerance = 10
    max_residual = 100
    while max_residual > tolerance:
        p_max_abs = np.polyfit(xy[:, 1], xy[:, 0], deg=1)
        residuals = xy[:, 0] - np.polyval(p_max_abs, xy[:, 1])
        max_residual = np.abs(residuals).max()
        if max_residual > tolerance:
            xy = np.delete(xy, np.argmax(np.abs(residuals)), axis=0)

    for index in range(nrows):
        curve = np.nanmean(normalized_image[rows * index: rows * (index+1), sides: -sides], axis=0)
        curve_filt = gaussian_filter1d(curve, 11)
        diff = (curve_filt[1:] - curve_filt[:-1]) / 2
        second_diff = -1 * (diff[1:] - diff[:-1]) / 2
        # peak of second derivative negative
        peaks_results = find_peaks(second_diff, width=10, prominence=0.00003)
        peaks_second_diff = peaks_results[0]
        properties_second_diff = peaks_results[1]
        y = (rows * index + rows * (index+1)) / 2
        x = y*p_max_abs[0] + p_max_abs[1] - sides

        indices = peaks_second_diff < x
        peaks_second_diff = np.delete(peaks_second_diff, indices)

        values = np.delete(properties_second_diff['right_ips'], indices)
        peak_second_diff = peaks_second_diff[0] + 1
        value = peaks_second_diff[0] + 1
        #fig, axes = plt.subplots(2, 1, sharex=True)
        #axes[0].plot([x, x], [np.nanmin(curve), np.nanmax(curve)], label='Absorption Max', color=[0, 1, 0])
        #axes[0].plot([value, value], [np.nanmin(curve), np.nanmax(curve)], label='Absorption Min', color=[1, 0, 0])
        #axes[0].plot(curve)
        #axes[0].plot(curve_filt)
        #axes[1].plot([x, x], [np.nanmin(second_diff), np.nanmax(second_diff)], label='Absorption Max', color=[0, 1, 0])
        #axes[1].plot([value, value], [np.nanmin(second_diff), np.nanmax(second_diff)], label='Absorption Min', color=[1, 0, 0])
        #axes[1].plot(second_diff)
        #axes[1].legend()
        #axes[0].set_title('Average along slow axis\n100 rows')
        #axes[1].set_title('Second derivative')
        #axes[1].set_xlabel('Pixels along fast axis')
        #plt.show()
        xy_min.append([
            sides + value, (rows * index + rows * (index+1)) / 2
            ])
    xy_min = np.array(xy_min)
    tolerance = 10
    max_residual = 100
    while max_residual > tolerance:
        p_min_abs = np.polyfit(xy_min[:, 1], xy_min[:, 0], deg=1)
        residuals = xy_min[:, 0] - np.polyval(p_min_abs, xy_min[:, 1])
        max_residual = np.abs(residuals).max()
        if max_residual > tolerance:
            xy_min = np.delete(xy_min, np.argmax(np.abs(residuals)), axis=0)


    return p_max_abs, p_min_abs, xy, xy_min


mpl.rc_file('matplotlibrc.txt')

angle = 0. * np.pi/180
h = 0.05
f = 1.5
t = 0.025
pixel_size = 0.177083333333

data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage()
normalized_image[mask] = np.nan
Y = MOD.Gradient(s, normalized_image, mask)
normalized_image = normalized_image / Y
p_max_abs, p_min_abs, xy_max, xy_min = FindMaxAbs(normalized_image)

y = np.array([0, 1920])
x_max = y*p_max_abs[0] + p_max_abs[1]
x_min = y*p_min_abs[0] + p_min_abs[1]
"""
fig, axes = plt.subplots(1, 2)
axes[0].imshow(data, vmin=0, vmax=5000)
axes[1].imshow(normalized_image, vmin=0, vmax=2)
axes[1].plot(xy_max[:, 0], xy_max[:, 1], linestyle='none', marker='.', color=[1, 0, 0])
axes[1].plot(x_max, y, color=[1, 0, 0])
axes[1].plot(xy_min[:, 0], xy_min[:, 1], linestyle='none', marker='.', color=[0, 1, 0])
axes[1].plot(x_min, y, color=[0, 1, 0])
plt.show()
"""
fig, axes = plt.subplots(1, 1)
im = axes.imshow(normalized_image, vmin=0.85, vmax=1.05)
axes.plot(xy_max[:, 0], xy_max[:, 1], linestyle='none', marker='.', color=[0, 1, 0])
axes.plot(x_max, y, color=[0, 1, 0], linewidth=1)
axes.plot(xy_min[:, 0], xy_min[:, 1], linestyle='none', marker='.', color=[1, 0, 0])
axes.plot(x_min, y, color=[1, 0, 0], linewidth=1)
axes.set_xticks([])
axes.set_yticks([])
fig.colorbar(im, ax=axes)
fig.tight_layout()
plt.show()

angle_max = np.arctan(p_max_abs[0])
angle_min = np.arctan(p_min_abs[0])

angle = (angle_max + angle_min) / 2

"""
m = (1/p_max_abs[0] + 1/p_min_abs[0]) / 2
b_max = -p_max_abs[1] / p_max_abs[0]
b_min = -p_min_abs[1] / p_min_abs[0]
f = t * 124 * np.sqrt(1 + m**2) / (b_max - b_min)
h = t * b_min / (b_max - b_min)
"""

b_min = -1 * pixel_size * (p_min_abs[1] - 960)
b_max = -1 * pixel_size * (p_max_abs[1] - 960)
h = t / (1 - b_min/b_max)
f = h * 124 / (b_min * np.cos(angle))

'''
R_z = np.array((
    (np.cos(angle), -np.sin(angle)),
    (np.sin(angle), np.cos(angle))
    ))
R_z_90 = np.array((
    (0, -1),
    (1, 0)
    ))
n_hat = np.matmul(R_z, np.array((0, -1)))
orthogonal_distance = np.matmul(s[:, :2], n_hat)
parallel_distance = np.matmul(s[:, :2], np.matmul(R_z_90, n_hat))

region = 0.4*detector_shape[0]*pixel_size
distance = np.linspace(
    -region,
    region,
    200)
distance_centers = (distance[1:] + distance[:-1]) / 2
mask_here = mask.copy()
mask_here[960-100: 960+100, :] = 1
counts = np.histogram(
    orthogonal_distance[np.invert(mask_here).flatten()],
    distance
    )
summed = np.histogram(
    orthogonal_distance[np.invert(mask_here).flatten()],
    distance,
    weights=normalized_image[np.invert(mask_here)].flatten()
    )
n_parallel_bins = 8
parallel_bins = np.linspace(-region, region, n_parallel_bins + 1)
averaged = np.zeros((n_parallel_bins + 1, distance_centers.size))
averaged[0, :] = summed[0] / counts[0]
for index in range(n_parallel_bins):
    indices = np.logical_and(
        parallel_distance > parallel_bins[index],
        parallel_distance <= parallel_bins[index + 1],
        )
    counts = np.histogram(
        orthogonal_distance[indices][np.invert(mask).flatten()[indices]],
        distance
        )
    summed = np.histogram(
        orthogonal_distance[indices][np.invert(mask).flatten()[indices]],
        distance,
        weights=normalized_image.flatten()[indices][np.invert(mask).flatten()[indices]]
        )
    averaged[index + 1, :] = summed[0] / counts[0]

"""
fig, axes = plt.subplots(1, 1)
for index in range(n_parallel_bins + 1):
    axes.plot(distance_centers, averaged[index, :], label=index)
axes.legend()
plt.show()
"""
"""
fig, axes = plt.subplots(1, 2)
axes[0].imshow(distance_array.reshape(detector_shape), origin='lower')
axes[1].imshow(normalized_image, origin='lower', vmin=0, vmax=2)
"""


maximum_distance = distance_centers[np.argmin(averaged[0, :])]

peaks = find_peaks(1 / averaged[0, :], width=5, prominence=0.01)[0]
maximum_distance = distance_centers[peaks[0]]
ratio = np.abs(maximum_distance / s[0, 2])
f = 0.7
t = 0.05
h_opt = ratio * f - t
print(h_opt)
absorption_opt = GetModel(angle, h_opt, f, t, s_norm)

    
fig, axes = plt.subplots(1, 1)
axes.plot(
    distance_centers, averaged[0, :] / averaged[0, :].max(),
    label='Data'
    )
axes.plot(
    pixel_size * (np.arange(detector_shape[1]) - 960),
    absorption.mean(axis=0),
    label='Guess'
    )
axes.plot(
    pixel_size * (np.arange(detector_shape[1]) - 960),
    absorption_opt.mean(axis=0),
    label='Optimized'
    )
axes.plot(
    pixel_size * (np.arange(detector_shape[1]) - 960),
    gaussian_filter1d(absorption_opt.mean(axis=0), 20),
    label='Optimized - blurred'
    )
axes.set_title(str(index))
axes.legend()
plt.show()
    
fig, axes = plt.subplots(2, 4)
axes[0, 0].imshow(absorption, origin='lower')
axes[0, 1].imshow(normalized_image, origin='lower', vmin=0, vmax=2)
axes[0, 2].imshow(normalized_image / absorption, origin='lower', vmin=0, vmax=2)
axes[0, 3].imshow(normalized_image / absorption_opt, origin='lower', vmin=0, vmax=2)
axes[1, 0].imshow(absorption, origin='lower')
axes[1, 1].imshow(data, origin='lower', vmin=0, vmax=5000)
axes[1, 2].imshow(data / absorption, origin='lower', vmin=0, vmax=5000)
axes[1, 3].imshow(data / absorption_opt, origin='lower', vmin=0, vmax=5000)
plt.show()
'''