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


def Fit(params, t, normalized_image, s_norm, mask, kapton_absorption_length):
	angle = params[0]
	h = params[1]
	f = params[2]
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	residuals = (normalized_image - absorption)[np.invert(mask)]
	return np.linalg.norm(residuals)


def FitWater(params, t, normalized_image, s_norm, mask, kapton_absorption_length):
	angle = params[0]
	h = params[1]
	f = params[2]
	v = params[3]
	r_drop = params[4]
	scale = params[5]
	absorption_water = MOD.WaterAbsorption(angle, h, f, t, s_norm, v, r_drop)
	
	#print(absorption_water)
	absorption_kapton = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	absorption = scale * absorption_water * absorption_kapton
	residuals = (normalized_image - absorption)[np.invert(mask)]
	print()
	print(f)
	print(h)
	print(v)
	print(r_drop)
	print(scale)
	print(np.linalg.norm(residuals))
	if np.isnan(absorption).sum() == 0:
		return np.linalg.norm(residuals)
	else:
		return 1000000


def FitScalar(f, angle, h, t, normalized_image, s_norm, kapton_absorption_length):
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	residuals = normalized_image - absorption
	return np.linalg.norm(residuals)


def GetScale(normalized_image):
	return normalized_image[:, 1000:].mean()


mpl.rc_file('matplotlibrc.txt')

angle = 0.55 * np.pi/180
h = 0.04
f = 0.665

#angle = 0.55 * np.pi/180
#h = 0.045
#f = 1.29
t = 0.025

data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=0.9)
normalized_image[mask] = np.nan

"""
Y = MOD.Gradient(s, normalized_image, mask)
normalized_image = normalized_image / Y

results_fit = minimize(
	Fit,
	x0=(angle, h, f),
	args=(t, normalized_image, s_norm, mask, kapton_absorption_length),
	method='L-BFGS-B',
	bounds=((-np.pi, np.pi), (0.001, None), (0.001, None))
	)
angle_fit = results_fit.x[0]
h_fit = results_fit.x[1]
f_fit = results_fit.x[2]
print(results_fit)
"""
results_fit = minimize(
	FitWater,
	x0=(angle, h, f, 0, 0.15, 1),
	args=(t, normalized_image, s_norm, mask, kapton_absorption_length),
	method='L-BFGS-B',
	bounds=((-np.pi, np.pi), (0.001, 0.1), (0.001, None), (-0.05, 0.05), (0.1, 0.2), (0, 2))
	)
angle_fit = results_fit.x[0]
h_fit = results_fit.x[1]
f_fit = results_fit.x[2]
v_fit = results_fit.x[3]
r_fit = results_fit.x[4]
scale = results_fit.x[5]
print(results_fit)

"""

results_fit = minimize_scalar(
	FitScalar,
	args=(angle, h, t, normalized_image, s_norm, kapton_absorption_length),
	method='Bounded',
	bounds=(0.001, 2)
	)
h_fit = h
f_fit = results_fit.x
print(results_fit)
"""

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