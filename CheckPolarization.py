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

mpl.rc_file('matplotlibrc.txt')

#p_frac = np.linspace(0, 1, 2)
p_frac = np.array([0, 0.8, 0.85, 0.9, 0.95, 1.0])
#p_frac = np.array([0, 1.0])
data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=0.92)
phi = -1*np.arctan2(s[:, 0], s[:, 1]) - np.pi/2
phi[phi<0] += 2*np.pi
#phi += 3*np.pi/2
phi_int = np.linspace(0, 2*np.pi, 32)

R = np.sqrt(s[:, 0]**2 + s[:, 1]**2)
theta2_array = np.arctan(R / np.abs(s[:, 2]))
theta2_indices = np.logical_and(
	theta2_array >= 10 * np.pi / 180,
	theta2_array <= 53 * np.pi/180
	)
integratation_indices = np.logical_and(
	theta2_indices.reshape((1920, 1920)),
	np.invert(mask)
	)
integration_counts = np.histogram(
		phi[integratation_indices.flatten()],
		bins=phi_int
		)
bins = integration_counts[1]
bin_centers = (bins[1:] + bins[:-1])/2

integrated = np.zeros((p_frac.size, bin_centers.size))


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes[0].imshow(normalized_image, vmin=0.8, vmax=1.2)
im = axes[1].imshow(phi.reshape((1920, 1920)))
axes[2].imshow(integratation_indices.reshape((1920, 1920)))
axes[0].set_title('Normalized Image')
axes[1].set_title('Phi for plotting')
axes[2].set_title('Integration indices')
fig.colorbar(im, ax=axes[1])
for index in range(3):
	axes[index].set_yticks([])
	axes[index].set_xticks([])

fig, axes = plt.subplots(1, 1, figsize=(6,5))
for index, p in enumerate(p_frac):
	data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=p)
	integration_sum = np.histogram(
		phi[integratation_indices.flatten()],
		bins=phi_int,
		weights=normalized_image[integratation_indices.reshape((1920, 1920))].flatten()
		)
	integrated[index, :] = integration_sum[0] / integration_counts[0]
	axes.plot(bin_centers, integrated[index, :], label='%1.2f'%(p))
axes.legend(loc='upper right', labelspacing=0.1, fontsize=14, title='Polarization\nFraction')
axes.set_xlabel('Phi')
axes.set_ylabel('Intensity')
fig.tight_layout()
plt.show()