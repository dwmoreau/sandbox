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


def GetModel(angle, h, f, t, s_norm, out='absorption'):
	# https://henke.lbl.gov/optical_constants/atten2.html
	# absorption at 10 keV
	n1 = -h * np.array((0, 1, 0)).T 
	n2 = -(h + t) * np.array((0, 1, 0)).T
	n3 = f * np.array((0, 0, 1)).T

	Rz = np.array((
		(np.cos(angle), -np.sin(angle), 0),
		(np.sin(angle), np.cos(angle), 0),
		(0, 0, 1)
		))

	n1 = np.matmul(Rz, n1)
	n2 = np.matmul(Rz, n2)
	n3 = np.matmul(Rz, n3)

	L1 = np.linalg.norm(n1)**2 / np.matmul(s_norm, n1)
	L2 = np.linalg.norm(n2)**2 / np.matmul(s_norm, n2)
	L3 = np.linalg.norm(n3)**2 / np.matmul(s_norm, n3)

	indices1 = np.logical_and(
		L1 > L3,
		L1 >= 0
		)
	indices2 = np.logical_and(
		L1 < L3,
		L2 >= L3
		)
	indices3 = np.logical_and(
		L3 > L2 ,
		L2 >= 0
		)

	path_length = np.zeros(s_norm.shape[:2])
	path_length[indices2] = L3[indices2] - L1[indices2]
	path_length[indices3] = L2[indices3] - L1[indices3]
	if out == 'absorption':
		return GetAbsorption(path_length)
	elif out == 'path_length':
		return path_length


def GetAbsorption(path_length):
	kapton_absorption_length = 2.34804
	return np.exp(-path_length / kapton_absorption_length)


def IntegrateModel(angle, h, f, t, s_norm, f_range, n=5):
	path_length_int = np.zeros((s_norm.shape[0], s_norm.shape[1], n))
	f_int = np.linspace(f - f_range, f + f_range, n)
	df = f_int[1] - f_int[0]
	#fig, axes = plt.subplots(2, 1, figsize=(10, 20))
	for index, f_here in enumerate(f_int):
		path_length_int[:, :, index] = GetModel(angle, h, f_here, t, s_norm, out='path_length')
		#axes[0].plot(path_length_int[960, :, index], label='%1.2f'%(f_here))
		#axes[1].plot(GetAbsorption(path_length_int[960, :, index]), label='%1.2f'%(f_here))
		#expected_position = 960 - 124/pixel_size * (h+t)/f_here
		#print(expected_position)
	path_length = np.trapz(path_length_int, f_int, axis=2) / (f_int[-1] - f_int[0])
	#axes[0].plot(path_length[960, :], linestyle=':', color=0 * np.ones(3))
	#axes[1].plot(GetAbsorption(path_length[960, :]), linestyle=':', color=0 * np.ones(3))
	#plt.legend()
	#plt.show()
	return GetAbsorption(path_length)


def NormalizeImage(data, s, mask):
	R = np.sqrt(s[:, 0]**2 + s[:, 1]**2)
	theta2_array = np.arctan(R / np.abs(s[:, 2]))
	theta2_int = np.linspace(3.5, 60, 100) * np.pi/180

	phi = np.pi - 1*np.arctan2(s[:, 0], s[:, 1])
	f = 0.85
	polarization = (1 - f*np.cos(2*phi)*np.sin(theta2_array)**2 / (1+np.cos(theta2_array)**2))
	data_polarization_corrected = data / polarization.reshape(data.shape)
	integration_counts = np.histogram(
		theta2_array[np.invert(mask).flatten()],
		bins=theta2_int
		)
	integration_sum = np.histogram(
		theta2_array[np.invert(mask).flatten()],
		bins=theta2_int,
		weights=data_polarization_corrected[np.invert(mask)].flatten()
		)
	bins = integration_counts[1]
	bin_centers = (bins[1:] + bins[:-1])/2
	integrated = integration_sum[0] / integration_counts[0]
	indices = np.invert(np.isnan(integrated))

	interpolated = np.interp(bin_centers, bin_centers[indices], integrated[indices])
	integrated_image = np.interp(theta2_array, bin_centers[indices], integrated[indices])

	normalized_image = data_polarization_corrected / integrated_image.reshape(data.shape)
	return normalized_image


def FindMaxAbs(normalized_image):
	rows = 100
	nrows = np.int(normalized_image.shape[0] / rows)
	sides = 10
	xy = []
	fig, axes = plt.subplots(1, 1)
	for index in range(nrows):
		curve = normalized_image[rows * index: rows * (index+1), sides: -sides].mean(axis=0)
		axes.plot(curve)
		filtered_inverse = gaussian_filter1d(1/curve, 1)
		peaks = find_peaks(filtered_inverse, width=30, prominence=0.05)[0]
		for peak in peaks:
			xy.append([
				sides + peak, (rows * index + rows * (index+1)) / 2
				])
	plt.show()
	xy = np.array(xy)
	tolerance = 10
	max_residual = 100
	while max_residual > tolerance:
		p_max_abs = np.polyfit(xy[:, 1], xy[:, 0], deg=1)
		residuals = xy[:, 0] - np.polyval(p_max_abs, xy[:, 1])
		max_residual = np.abs(residuals).max()

		if max_residual > tolerance:
			xy = np.delete(xy, np.argmax(np.abs(residuals)), axis=0)
	return p_max_abs, xy


angle = 0 * np.pi/180
h = 0.05
f = 1.5
t = 0.05
pixel_size = 0.177083333333

# Load image
image_file_name = '/home/david/dials_dev/build/dials_data/lcls_rayonix_kapton/hit-20181213155134902.cbf'
image = dxtbx.load(image_file_name)
panel = image.get_detector()[0]
data = image.get_raw_data().as_numpy_array()
mask = np.logical_or(
	data == 0,
	data > 5000
	)

# Calculate vector from the crystal to the pixels
x1 = y1 = 0
detector_shape = panel.get_image_size()
x, y = np.meshgrid(
	np.linspace(x1, detector_shape[0] - 1, detector_shape[0] - x1),
	np.linspace(y1, detector_shape[1] - 1, detector_shape[1] - y1)
	)
mm = panel.pixel_to_millimeter(flex.vec2_double(flex.double(y.flatten()), flex.double(x.flatten())))

s = panel.get_lab_coord(mm).as_numpy_array()
s_norm = (s.T / np.linalg.norm(s, axis=1)).T.reshape((detector_shape[0], detector_shape[1], 3))
s_norm[:, :, 1] *= -1
s_norm[:, :, 2] *= -1

# Calculations
absorption = IntegrateModel(angle, h, f, t, s_norm, 0.2, n=3)

normalized_image = NormalizeImage(data, s, mask)
normalized_image[mask] = 0
p_max_abs, xy = FindMaxAbs(normalized_image)

"""
y = np.array([0, 1920])
x = y*p_max_abs[0] + p_max_abs[1]
fig, axes = plt.subplots(1, 2)
axes[0].imshow(data, origin='lower', vmin=0, vmax=5000)
axes[1].imshow(normalized_image, origin='lower', vmin=0, vmax=2)
axes[1].plot(xy[:, 0], xy[:, 1], linestyle='none', marker='.', color=[1, 0, 0])
axes[1].plot(x, y)
plt.show()
"""

angle = np.arctan(p_max_abs[0])
print(angle * 180/np.pi)
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


fig, axes = plt.subplots(1, 1)
for index in range(n_parallel_bins + 1):
	axes.plot(distance_centers, averaged[index, :], label=index)
axes.legend()
plt.show()

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
#axes.plot(
#	pixel_size * (np.arange(detector_shape[1]) - 960),
#	gaussian_filter1d(absorption_opt.mean(axis=0), 20),
#	label='Optimized - blurred'
#	)
axes.set_title(str(index))
axes.legend()
plt.show()

"""
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
"""
