"""
Issues:
	1: Normalizing an image with large amounts of zeros
	
	2: how to best handle the data - logistically
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
import matplotlib.pyplot as plt
import numpy as np


image_file_name = '/home/david/Documents/Background/Data/run_000795.JF07T32V01_master.h5'
polarization_fraction = 1

theta2_int = np.linspace(0, 40, 100) * np.pi/180
phi_int = np.linspace(0, 2*np.pi, 16)
image = dxtbx.load(image_file_name)
beam = image.get_beam()
wavelength = beam.get_wavelength()
data = image.get_raw_data()
detector = image.get_detector()

"""
need for each panel
	I
	theta2
	phi
	mask
	mm
	s
	s_norm
"""
keys = ['I', 'I_flat', 'I_norm', 'theta2', 'phi', 'mask', 'mm', 's', 's_norm', 'polarization', 'az_avg']
panels = [dict.fromkeys(keys) for i in range(8)]
flat = [dict.fromkeys(keys) for i in range(8)]

fig_im, axes_im = plt.subplots(4, 2)
fig_im_norm, axes_im_norm = plt.subplots(4, 2)
fig_az, axes_az = plt.subplots(1, 1)
fig_phi, axes_phi = plt.subplots(1, 1)
"""
fig_fast, axes_fast = plt.subplots(4, 2)
fig_slow, axes_slow = plt.subplots(4, 2)
fig_theta2, axes_theta2 = plt.subplots(4, 2)
fig_phi, axes_phi = plt.subplots(4, 2)
"""
row_column = [
	[0, 0],
	[1, 0],
	[2, 0],
	[3, 0],
	[0, 1],
	[1, 1],
	[2, 1],
	[3, 1]
	]
detector_shape = detector[0].get_image_size()
extent = [
	[detector_shape[1], 0, detector_shape[0], 0],
	[detector_shape[1], 0, detector_shape[0], 0],
	[detector_shape[1], 0, detector_shape[0], 0],
	[detector_shape[1], 0, detector_shape[0], 0],
	[0, detector_shape[1], 0, detector_shape[0]],
	[0, detector_shape[1], 0, detector_shape[0]],
	[0, detector_shape[1], 0, detector_shape[0]],
	[0, detector_shape[1], 0, detector_shape[0]],
	]
	#(left, right, bottom, top)
intensities = np.zeros((detector_shape[0]*detector_shape[1], 8, 2))
for index in range(8):
	row, column = row_column[index]
	panel = detector[index]
	panels[index]['I'] = data[index].as_numpy_array().swapaxes(0, 1)
	panels[index]['mask'] = np.logical_or(
		panels[index]['I'] < 0,
		panels[index]['I'] > 40
		)
	panels[index]['I'][panels[index]['mask']] = -1
	flat[index]['I'] = panels[index]['I'].ravel()
	flat[index]['mask'] = panels[index]['mask'].ravel()
	origin = panel.get_origin()
	fast_axis = panel.get_fast_axis()
	slow_axis = panel.get_slow_axis()

	axes_im[row, column].set_title(
		'origin: %1.2f, %1.2f, %1.2f\nfast: %1.2f, %1.2f, %1.2f\n, slow: %1.2f, %1.2f, %1.2f\n'
		% (*origin, *fast_axis, *slow_axis), fontsize=10
		)
	
	# Calculate vector from the crystal to the pixels
	x1 = y1 = 0
	
	x, y = np.meshgrid(
		np.linspace(x1, detector_shape[1] - 1, detector_shape[1] - x1),
		np.linspace(y1, detector_shape[0] - 1, detector_shape[0] - y1)
		)
	
	flat[index]['mm'] = panel.pixel_to_millimeter(flex.vec2_double(flex.double(y.flatten()), flex.double(x.flatten())))
	flat[index]['s'] = panel.get_lab_coord(flat[index]['mm']).as_numpy_array()
	panels[index]['s_norm'] = (flat[index]['s'].T / np.linalg.norm(flat[index]['s'], axis=1)).T.reshape((detector_shape[0], detector_shape[1], 3))
	panels[index]['s_norm'][:, :, 1] *= -1
	panels[index]['s_norm'][:, :, 2] *= -1
	R = np.sqrt(flat[index]['s'][:, 0]**2 + flat[index]['s'][:, 1]**2)
	flat[index]['theta2'] = np.arctan(R / np.abs(flat[index]['s'][:, 2]))
	flat[index]['phi'] = np.pi - 1*np.arctan2(flat[index]['s'][:, 0], flat[index]['s'][:, 1])
	flat[index]['polarization'] = (1 - polarization_fraction*np.cos(2*flat[index]['phi'])*np.sin(flat[index]['theta2'])**2 / (1+np.cos(flat[index]['theta2'])**2))
	

	integration_sum = np.histogram(
		flat[index]['theta2'][np.invert(flat[index]['mask'])],
		bins=theta2_int,
		weights=flat[index]['I'][np.invert(flat[index]['mask'])] / flat[index]['polarization'][np.invert(flat[index]['mask'])]
		)
	integration_counts = np.histogram(
		flat[index]['theta2'][np.invert(flat[index]['mask'])],
		bins=theta2_int
		)
	bins = integration_counts[1]
	bin_centers = (bins[1:] + bins[:-1])/2
	integrated = integration_sum[0] / integration_counts[0]
	axes_az.plot(180/np.pi * bin_centers, integrated)

	indices = np.invert(np.isnan(integrated))
	flat[index]['az_avg'] = np.interp(flat[index]['theta2'], bin_centers[indices], integrated[indices])
	#flat[index]['I_norm'] = flat[index]['I'] / (flat[index]['az_avg'] * flat[index]['polarization'])
	flat[index]['I_norm'] = ((flat[index]['I'] / flat[index]['polarization']) - flat[index]['az_avg']) / flat[index]['az_avg']
	panels[index]['I_norm'] = flat[index]['I_norm'].reshape((detector_shape[0], detector_shape[1]))
	intensities[:, index, 0] = flat[index]['I']
	intensities[:, index, 1] = flat[index]['I_norm']
	if column == 0:
		axes_im[row, column].imshow(panels[index]['I'][:, ::-1], vmin=0, vmax=20,
			extent=extent[index]
			)
		axes_im_norm[row, column].imshow(panels[index]['I_norm'][:, ::-1], vmin=-1, vmax=1,
			extent=extent[index]
			)
	elif column == 1:
		axes_im[row, column].imshow(panels[index]['I'][::-1, :], vmin=0, vmax=20,
			extent=extent[index]
			)
		axes_im_norm[row, column].imshow(panels[index]['I_norm'][::-1, :], vmin=-1, vmax=1,
			extent=extent[index]
			)

	phi_sum = np.histogram(
		flat[index]['phi'][np.invert(flat[index]['mask'])],
		bins=phi_int,
		weights=flat[index]['I_norm'][np.invert(flat[index]['mask'])]
		)
	phi_counts = np.histogram(
		flat[index]['phi'][np.invert(flat[index]['mask'])],
		bins=phi_int
		)

	bins = phi_counts[1]
	bin_centers = (bins[1:] + bins[:-1])/2
	integrated = phi_sum[0] / phi_counts[0]
	axes_phi.plot(bin_centers, integrated)
	
	"""
	#axes[row, column].imshow(panels[index]['phi'].reshape((detector_shape[1], detector_shape[0])))
	#axes[row, column].imshow(panels[index]['s_norm'][:, :, 0])
	panels[index]['s'] = panels[index]['s'].reshape((detector_shape[0], detector_shape[1], 3))
	panels[index]['theta2'] = panels[index]['theta2'].reshape((detector_shape[0], detector_shape[1]))
	panels[index]['phi'] = panels[index]['phi'].reshape((detector_shape[0], detector_shape[1]))
	if column == 0:
		axes_fast[row, column].imshow(
			panels[index]['s'][:, ::-1, 0], vmin=-30, vmax=30,
			extent=extent[index]
			)
		axes_slow[row, column].imshow(
			panels[index]['s'][:, ::-1, 1], vmin=-30, vmax=30,
			extent=extent[index]
			)
		axes_theta2[row, column].imshow(
			180/np.pi*panels[index]['theta2'][:, ::-1],
			extent=extent[index], vmin=0, vmax=40
			)
		axes_phi[row, column].imshow(
			panels[index]['phi'][:, ::-1],
			extent=extent[index], vmin=0, vmax=2*np.pi
			)
	elif column == 1:
		axes_fast[row, column].imshow(
			panels[index]['s'][::-1, :, 0], vmin=-50, vmax=50,
			extent=extent[index]
			)
		axes_slow[row, column].imshow(
			panels[index]['s'][::-1, :, 1], vmin=-50, vmax=50,
			extent=extent[index]
			)
		axes_theta2[row, column].imshow(
			180/np.pi*panels[index]['theta2'][::-1, :],
			extent=extent[index], vmin=0, vmax=38
			)
		axes_phi[row, column].imshow(
			panels[index]['phi'][::-1, :],
			extent=extent[index], vmin=0, vmax=2*np.pi
			)
	"""
"""
def Assemble(panels, key):
	assembled = np.block([
		[panels[0][key][:, ::-1], panels[4][key][::-1, :]],
		[panels[1][key][:, ::-1], panels[5][key][::-1, :]],
		[panels[2][key][:, ::-1], panels[6][key][::-1, :]],
		[panels[3][key][:, ::-1], panels[7][key][::-1, :]]
		])
	return assembled

intensities = Assemble(panels, 'I')
"""
bins_I = np.linspace(0, 30, 31)
bins_I_norm = np.linspace(-5, 5, 101)
bin_centers_I = (bins_I[1:] + bins_I[:-1]) / 2
bin_centers_I_norm = (bins_I_norm[1:] + bins_I_norm[:-1]) / 2
indices = intensities[:, :, 0] >= 0
hist_I = np.histogram(intensities[:, :, 0][indices], bins=bins_I)
hist_I_norm = np.histogram(intensities[:, :, 1][indices], bins=bins_I_norm)
fig, axes = plt.subplots(1, 2)
axes[0].bar(bin_centers_I, hist_I[0], width=bin_centers_I[1]-bin_centers_I[0])
axes[1].bar(bin_centers_I_norm, hist_I_norm[0], width=bin_centers_I_norm[1]-bin_centers_I_norm[0])
plt.show()
