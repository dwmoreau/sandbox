"""
Why is there a sharp downward turn at the ends of the image
	1 or 2 pixels at the detector edges are either 0 or sharply smaller
Why does the flattened image decrease on the left side?
	Polarization correction - fraction of polarized x-rays was too small - need to use 0.95 to get this flat
Why is the maximum absorption too large?
	larger h makes the maximum absorption smaller
"""

import sys
sys.path.append('/home/david/dials_dev/modules/dxtbx/src')
sys.path.append('/home/david/dials_dev/build/lib')
sys.path.append('/home/david/dials_dev/modules')

from cctbx import factor_kev_angstrom
from dials.array_family import flex
import dxtbx
#from dxtbx.model.experiment_list import ExperimentListFactory
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import Modules as MOD


def NormalizeImage(data, s, mask, polarization_fraction):
	# Flattens the image - correcting for polarization and radially symmetric 
	# scattering

	# Set up for azimuthal averaging
	# This assumes that the beam direction is (0, 0, -1)
	R = np.sqrt(s[:, 0]**2 + s[:, 1]**2)
	theta2_array = np.arctan(R / np.abs(s[:, 2]))
	theta2_int = np.linspace(3.5, 65, 100) * np.pi/180

	# Polarization correction
	# For more information see the following papers:
	#	Kahn, R. et al. (1982), J. Appl. Cryst. 15, 330-337
	#	Sulyanov, S. et al. (2014), J. Appl. Cryst. 47, 1449-1451
	# polarization fraction is the fraction of x-rays polarized in the
	# orientation of the monochromator crystal
	phi = np.pi - 1*np.arctan2(s[:, 0], s[:, 1])
	polarization = (1 - polarization_fraction*np.cos(2*phi)*np.sin(theta2_array)**2 / (1+np.cos(theta2_array)**2))
	data_polarization_corrected = data / polarization.reshape(data.shape)
	
	#fig, axes = plt.subplots(1, 1)
	#axes.imshow(phi.reshape(data.shape))
	#plt.show()

	# The image quadrant with phi <= pi is unaffected by kapton absorption
	# and odd, unexplained high scattering intensity. This should not be
	# used in the final version of the code
	#phi_indices = np.logical_or(
	#	phi <= np.pi / 2,
	#	phi >= 3 * np.pi / 2 
	#	)
	phi_indices = phi <= np.pi
	#phi_indices = np.ones(phi.shape)
	# These are the portions of the detector that are used to do the 
	# azimuthal average
	integratation_indices = np.logical_or(
		phi_indices.reshape(data.shape),
		np.invert(mask)
		)
	
	#integratation_indices = np.invert(mask)
	# This calculates the average scatter into a pixel at a given 2theta angle 
	# It works by counting the total number of photons scattered into all the 
	# pixels within a 2theta bin. Then dividing by the total number of pixels
	# in the 2theta bin
	integration_sum = np.histogram(
		theta2_array[integratation_indices.flatten()],
		bins=theta2_int,
		weights=data_polarization_corrected[integratation_indices].flatten()
		)
	integration_counts = np.histogram(
		theta2_array[integratation_indices.flatten()],
		bins=theta2_int
		)
	
	bins = integration_counts[1]
	bin_centers = (bins[1:] + bins[:-1])/2
	integrated = integration_sum[0] / integration_counts[0]

	# bins with zero pixels produce a divide by zero error resulting in nan
	indices = np.invert(np.isnan(integrated))

	# This expands the 1D azimuthal average to the 2D detector image
	#interpolated = np.interp(bin_centers, bin_centers[indices], integrated[indices])
	integrated_image = np.interp(theta2_array, bin_centers[indices], integrated[indices])
	
	normalized_image = data_polarization_corrected / integrated_image.reshape(data.shape)
	"""
	fig, axes = plt.subplots(1, 3, figsize=(12,5))
	axes[0].plot(180/np.pi * bin_centers, integrated)
	im1 = axes[1].imshow(data, vmin=0, vmax=5000)
	im2 = axes[2].imshow(data / integrated_image.reshape(data.shape), vmin=0.5, vmax=1.5)
	axes[1].set_xticks([])
	axes[1].set_yticks([])
	axes[1].set_title('Raw Data')
	fig.colorbar(im1, ax=axes[1])
	axes[2].set_xticks([])
	axes[2].set_yticks([])
	axes[2].set_title('Corrected for\nazimuthal scatter')
	fig.colorbar(im2, ax=axes[2])
	axes[0].set_ylabel('Intensity')
	axes[0].set_xlabel('2$\\theta$')
	fig.tight_layout()
	#plt.show()
	
	im = [[], [], []]
	fig, axes = plt.subplots(1, 3, figsize=(14, 5))
	im[0] = axes[0].imshow(polarization.reshape(data.shape))
	im[1] = axes[1].imshow(data / integrated_image.reshape(data.shape), vmin=0.5, vmax=1.5)
	im[2] = axes[2].imshow(normalized_image, vmin=0.5, vmax=1.5)
	for index in range(3):
		axes[index].set_xticks([])
		axes[index].set_yticks([])
		fig.colorbar(im[index], ax=axes[index])
	axes[0].set_title('Polarization Correction')
	axes[1].set_title('Normalized Image\nNo Polarization Correction')
	axes[2].set_title('Normalized Image\nWith Polarization Correction')
	fig.tight_layout()
	plt.show()
	"""
	return normalized_image


def Gradient(s, normalized_image, mask):
	phi = np.pi - 1*np.arctan2(s[:, 0], s[:, 1])
	phi_indices = np.logical_or(
		phi <= np.pi / 2,
		phi >= 3 * np.pi / 2 
		)
	fit_indices = np.logical_and(
		phi_indices,
		np.invert(mask).flatten()
		)
	x, y = np.meshgrid(
		np.arange(mask.shape[0]),
		np.arange(mask.shape[1])
		)
	A = np.vstack((
		x.flatten(),
		y.flatten(),
		np.ones(mask.shape).flatten()
		)).T
	plane_results = np.linalg.lstsq(A[fit_indices, :], normalized_image.flatten()[fit_indices])
	Y = np.matmul(A, plane_results[0]).reshape(mask.shape)
	"""
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))
	im = [[], [], []]
	im[0] = axes[0].imshow(Y)
	im[1] = axes[1].imshow(normalized_image, vmin=0.9, vmax=1.1)
	im[2] = axes[2].imshow(normalized_image / Y, vmin=0.9, vmax=1.1)
	for index in range(3):
		axes[index].set_yticks([])
		axes[index].set_xticks([])
		fig.colorbar(im[index], ax=axes[index])
	axes[0].set_title('Fit Gradient')
	axes[1].set_title('Normalized Image')
	axes[2].set_title('Normalized Image\nGradient Corrected')
	fig.tight_layout()
	plt.show()
	"""
	return Y


# From paper
# t = 0.050 mm
# width = 1/16" to 1/8"
# width = 1.6 to 3.2 mm
# 	=> f = 0.8 to 1.6
# Drop diameter ~ 0.3 mm
# h < 0.15

mpl.rc_file('matplotlibrc.txt')

angle = 0.55 * np.pi/180
h = 0.04
f = 0.665
t = 0.025
polarization_fraction = 0.9
data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage()

#normalized_image = MOD.NormalizeImage(data, s, mask, polarization_fraction)
normalized_image[mask] = np.nan
Y = MOD.Gradient(s, normalized_image, mask)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
im = [[], []]
im[0] = axes[0].imshow(data, vmin=0, vmax=5000)
im[1] = axes[1].imshow(normalized_image / Y, vmin=0.9, vmax=1.1)
for index in range(2):
	axes[index].set_yticks([])
	axes[index].set_xticks([])
	fig.colorbar(im[index], ax=axes[index])
axes[0].set_title('Raw Image')
axes[1].set_title('Normalized Image')
fig.tight_layout()

bins_I = np.linspace(0, 5000, 101)
bins_I_norm = np.linspace(0.5, 1.5, 101)
bin_centers_I = (bins_I[1:] + bins_I[:-1]) / 2
bin_centers_I_norm = (bins_I_norm[1:] + bins_I_norm[:-1]) / 2
hist_I = np.histogram(data[np.invert(mask)], bins=bins_I)
hist_I_norm = np.histogram(normalized_image[np.invert(mask)], bins=bins_I_norm)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].bar(bin_centers_I, hist_I[0], width=bin_centers_I[1]-bin_centers_I[0])
axes[1].bar(bin_centers_I_norm, hist_I_norm[0], width=bin_centers_I_norm[1]-bin_centers_I_norm[0])
fig.tight_layout()
plt.show()

"""
absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=1)
absorption_water = MOD.WaterAbsorption(angle, h, f, t, s_norm, v=0.05)
absorption_total = absorption * absorption_water

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im0 = axes[0].imshow(data, vmin=0, vmax=5000)
im1 = axes[1].imshow(normalized_image, vmin=0.8, vmax=1.2)
axes[2].plot(np.nanmean(normalized_image, axis=0) / 1.07, label='Normalized Image\n - scaled')
axes[2].plot(absorption.mean(axis=0), label='Absorption Model')
#axes[2].plot(absorption_water[1200:, :].mean(axis=0))
#axes[2].plot(absorption_total[1200:, :].mean(axis=0))
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xticks([])
axes[0].set_title('Raw Image')
axes[1].set_title('Normalized Image')
axes[2].set_title('Averaged along slow axis')
axes[2].set_xlabel('Pixels along fast axis')
axes[2].legend(loc='upper left')
axes[2].set_ylim([0.6, 1.15])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
axes[0].plot(np.nanmean(normalized_image, axis=0) / 1.07, label='Normalized Image\n - scaled')
axes[0].plot(absorption.mean(axis=0), label='Absorption Model')
axes[1].plot(np.nanmean(normalized_image, axis=0) / 1.07, label='Normalized Image')
axes[1].plot(
	np.nanmean(normalized_image / absorption, axis=0) / 1.07,
	label='Normalized Image\nCorrected For Absorption'
	)
fig.tight_layout()
axes[0].set_ylim([0.6, 1.3])
axes[0].legend(loc='upper left')
axes[1].legend(loc='lower left')
plt.show()
"""
"""
fig, axes = plt.subplots(1, 4, figsize=(15, 10), sharey=True)
for index in range(4):
	axes[index].plot(np.nanmean(normalized_image, axis=0) / 1.07, color=[0,0,0])
# Loop through f
for f_here in np.linspace(0.5, 2.5, 3):
	absorption = MOD.IntegrateModel(angle, h, f_here, t, s_norm, kapton_absorption_length, 0.1, n=3)
	axes[0].plot(absorption.mean(axis=0), label='%1.2f'%(f_here))
# Loop through h
for h_here in np.linspace(0.01, 0.15, 3):
	absorption = MOD.IntegrateModel(angle, h_here, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	axes[1].plot(absorption.mean(axis=0), label='%1.2f'%(h_here))
# Loop through t
for t_here in [0.025, 0.050, 0.075]:
	absorption = MOD.IntegrateModel(angle, h, f, t_here, s_norm, kapton_absorption_length, 0.1, n=3)
	axes[2].plot(absorption.mean(axis=0), label='%1.2f'%(t_here))
# Loop through kapton abs
for abs_len in [1, 2, 3]:
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, abs_len, 0.1, n=3)
	axes[3].plot(absorption.mean(axis=0), label='%1.2f'%(abs_len))

titles = ['f', 'h', 't', 'abs len']
for index in range(4):
	axes[index].legend()
	axes[index].set_title(titles[index])
axes[0].set_ylim([0.45, 1.1])
fig.tight_layout()
plt.show()


f_range = np.arange(0.7, 0.95, 0.05)
for f in f_range:
	normalized_image = NormalizeImage(data, s, mask, f)
	fig, axes = plt.subplots(1, 3, figsize=(12, 6))
	im0 = axes[0].imshow(data, vmin=0, vmax=5000)
	im1 = axes[1].imshow(normalized_image, vmin=0.8, vmax=1.2)
	axes[2].plot(normalized_image.mean(axis=0))
	fig.colorbar(im0, ax=axes[0])
	fig.colorbar(im1, ax=axes[1])
	axes[0].set_title(str(f))
	plt.show()

fig, axes = plt.subplots(1, 2)
bins = np.linspace(0, 5000, 101)
hist, bin_edges = np.histogram(data.flatten(), bins=bins)
bin_centers = (bins[1:] + bins[:-1]) / 2
axes[0].bar(bin_centers, hist, width=bins[1]-bins[0])
bins = np.linspace(0, 2, 101)
hist, bin_edges = np.histogram(normalized_image.flatten(), bins=bins)
bin_centers = (bins[1:] + bins[:-1]) / 2
axes[1].bar(bin_centers, hist, width=bins[1]-bins[0])
"""