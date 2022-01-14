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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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
		weights=data[integratation_indices].flatten()
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
	return data_polarization_corrected, integrated_image.reshape(data.shape)


def Fit(params, t, data_polarization_corrected, integrated_image, s_norm, mask, kapton_absorption_length):
	angle = params[0]
	h = params[1]
	f = params[2]
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	residuals = ((data_polarization_corrected - absorption*integrated_image) / integrated_image)[np.invert(mask)]
	print(angle)
	print(h)
	print(f)
	print(np.linalg.norm(residuals))
	print()
	return np.linalg.norm(residuals)


def GetScale(normalized_image):
	return normalized_image[:, 1000:].mean()


mpl.rc_file('matplotlibrc.txt')
polarization_fraction=0.9
angle = 0.55 * np.pi/180
h = 0.04
f = 0.665

#angle = 0.55 * np.pi/180
#h = 0.045
#f = 1.29
t = 0.025

data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=0.9)
data_polarization_corrected, integrated_image = NormalizeImage(data, s, mask, polarization_fraction)
normalized_image[mask] = np.nan


Y = MOD.Gradient(s, normalized_image, mask)
normalized_image = normalized_image / Y

results_fit = minimize(
	Fit,
	x0=(angle, h, f),
	args=(t, data_polarization_corrected, integrated_image*Y, s_norm, mask, kapton_absorption_length),
	method='L-BFGS-B',
	bounds=((-np.pi, np.pi), (0.001, None), (0.001, None))
	)
angle_fit = results_fit.x[0]
h_fit = results_fit.x[1]
f_fit = results_fit.x[2]
print(results_fit)

absorption_initial = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
absorption_fit = MOD.IntegrateModel(angle_fit, h_fit, f_fit, t, s_norm, kapton_absorption_length, 0.1, n=3)
"""
fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(data, vmin=0, vmax=5000)
axes[0, 1].imshow(absorption_initial)
axes[0, 2].imshow(absorption_fit)
axes[1, 0].imshow(normalized_image, vmin=0.8, vmax=1.2)
axes[1, 1].imshow(normalized_image / absorption_initial, vmin=0.8, vmax=1.2)
axes[1, 2].imshow(normalized_image / absorption_fit, vmin=0.8, vmax=1.2)

im = [[[], []], [[], []]]
fig, axes = plt.subplots(2, 2)
im[0][0] = axes[0, 0].imshow(data, vmin=0, vmax=5000)
im[0][1] = axes[0, 1].imshow(data / absorption_fit, vmin=0, vmax=5000)
im[1][0] = axes[1, 0].imshow(normalized_image, vmin=0.9, vmax=1.1)
im[1][1] = axes[1, 1].imshow(normalized_image / absorption_fit, vmin=0.9, vmax=1.1)
for row in range(2):
	for column in range(2):
		axes[row, column].set_yticks([])
		axes[row, column].set_xticks([])
		fig.colorbar(im[row][column], ax=axes[row, column])
axes[0, 0].set_title('Intial')
axes[0, 1].set_title('Corrected\nOptimization')
axes[0, 0].set_ylabel('Raw Image')
axes[1, 0].set_ylabel('Normalized Image')
fig.tight_layout()
plt.show()
"""
fig, axes = plt.subplots(1, 1, figsize=(8,5))
axes.plot(absorption_initial.mean(axis=0), label='Line Model')
axes.plot(absorption_fit.mean(axis=0), label='Optimized Model')
#axes.plot(np.nanmean(normalized_image / absorption_fit, axis=0), label='Corrected Image')
axes.plot(np.nanmean(normalized_image, axis=0), color=[0, 0, 0], label='Normalized Image')
axes.legend()
axes.set_title('Averaged along slow axis')
axes.set_xlabel('Pixels along fast axis')
fig.tight_layout()
plt.show()