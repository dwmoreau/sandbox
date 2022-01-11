"""
Correlate the inverse of absorption with the inverse of the image
Try fitting two lines to the curves across the 
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import Modules as MOD


def GetModel(angle, h, f, t, s_norm,  v=0, out='kapton'):
	# Returns the pathlength through the kapton film

	# s_norm: unit vector pointing from crystal to each detector pixel
	# n1: Vector pointing from crystal to front face of kapton
	# n2: Vector pointing from crystal to back face of kapton
	# n3: Vector pointing from crystal to far edge of kapton
	# delta: Vector point from the center of the water droplet on the kapton
	#	film to the center of the crystal
	n1 = -h * np.array((0, 1, 0)).T 
	n2 = -(h + t) * np.array((0, 1, 0)).T
	n3 = f * np.array((0, 0, 1)).T
	delta = h * np.array((0, 1, 0)).T + v * np.array((1, 0, 0)).T

	# Rotate these vectors to account for the kapton angle
	Rz = np.array((
		(np.cos(angle), -np.sin(angle), 0),
		(np.sin(angle), np.cos(angle), 0),
		(0, 0, 1)
		))

	n1 = np.matmul(Rz, n1)
	n2 = np.matmul(Rz, n2)
	n3 = np.matmul(Rz, n3)
	delta = np.matmul(Rz, delta)

	# Calculate the distance from the crystal to each kapton
	# face along each unit vector pointing to the pixels
	# These assume that the kapton faces are infinite planes
	L1 = np.linalg.norm(n1)**2 / np.matmul(s_norm, n1)
	L2 = np.linalg.norm(n2)**2 / np.matmul(s_norm, n2)
	L3 = np.linalg.norm(n3)**2 / np.matmul(s_norm, n3)

	# indices1: Paths that pass through the front face and the far edge
	# indices2: Paths that pass through the front face and the back face
	indices1 = np.logical_and(
		L1 < L3,
		L2 >= L3
		)
	indices2 = np.logical_and(
		L3 > L2 ,
		L2 >= 0
		)

	# Path length through kapton calculations
	path_length = np.zeros(s_norm.shape[:2])
	# For paths through the front face and the far edge
	path_length[indices1] = L3[indices1] - L1[indices1]
	# For paths through the front face and the back face
	path_length[indices2] = L2[indices2] - L1[indices2]

	# This calculates the path length through a water droplet
	# Assumes that the crystal is in a spherical droplet on the kapton film's
	# front face.
	# The crystal is a distance h from the kapton film along the n1 vector.
	#	This is the crystal height model from above.
	# The crystal is a distance v from the droplet's center along the vector
	#	along the kapton film.

	# n1 = unit vector from crystal to pixel
	# L4 = pathlength from crystal to water droplet surface along n1
	#	L4*n1 = vector from crystal to water droplet surface
	# delta = vector from center of water droplet to crystal
	#	delta = Rz * [h*(0, 1, 0) + v*(1, 0, 0)]
	# r = vector from center of water droplet on kapton film to water droplet
	#	surface such that r = delta + L4*n1 and |r| = r_drop.
	# To calculate L4, start with the relation
	# 	r*r  = (delta + L4*n1)*(delta + L4*n1)
	#	=> L4^2 + l4 * 2n1*delta + [|delta|^2 - |r|^2] = 0
	#	=> L4 = -delta*n1 + sqrt((delta*n1)^2 -(|delta|^2 - r_drop^2)
	
	r_drop = 0.1
	delta_snorm = np.matmul(s_norm, delta)
	delta_mag = np.linalg.norm(delta)
	L4 = -delta_snorm + np.sqrt(delta_snorm**2 - (delta_mag**2 - r_drop**2))

	# If x-ray path does not go through kapton film, the pathlength is equal to
	# the distance from the crystal to water droplet's surface.
	#	if L1 = 0 or L1 > L4:
	#		pathlength = L4
	# Otherwise, the kapton surface truncates the water pathlength and the 
	#	if L1 < L4 and L1 != 0:
	#		pathlength = L1
	indices_water1 = np.logical_and(
		L1 < L4,
		L1 >= 0
		)
	path_length_water = L4.copy()
	path_length_water[indices_water1] = L1[indices_water1]
	
	if out == 'kapton':
		return path_length
	elif out == 'water':
		return path_length_water
	elif out == 'both':
		return path_length, path_length_water


def WaterAbsorption(angle, h, f, t, s_norm, v):
	# Returns the absorption from water
	path_length_water = GetModel(angle, h, f, t, s_norm, v=v, out='water')
	water_absorption_length = 1.7 # at 9.5 kev or 1.3 A
	return MOD.GetAbsorption(path_length_water, water_absorption_length)


def IntegrateModelHF(angle, h, f, t, s_norm, kapton_absorption_length, f_range, h_range, n=3):
	# Returns the absorption from the kapton tape
	# If n == 1, calculate the absorption assuming all the scattering comes 
	# 	from a point
	# Else, calculates the absorption assuming a finite pathlength
	#	This is done by absorption = 1/L x integrate(absorption)
	#	If n == 3, the returned absorption is the average of the absorption model
	#	at the start, middle, and end of the pathlength through the water droplet
	#	n == 3 is a good choice - don't see any benefit on n > 5

	path_length = np.zeros((s_norm.shape[0], s_norm.shape[1]))
	f_int = np.linspace(f - f_range, f + f_range, n)
	h_int = np.linspace(h - h_range, h + h_range, n)
	df = f_int[1] - f_int[0]
	dh = h_int[1] - h_int[0]
	for f_index, f_here in enumerate(f_int):	
		for h_index, h_here in enumerate(h_int):
			index = f_index*n + h_index
			f_weight = 1
			h_weight = 1
			if f_index == 0 or f_index == n-1:
				f_weight = 1/2		
			if h_index == 0 or h_index == n-1:
				h_weight = 1/2
			weight = f_weight * h_weight
			path_length[:, :] += weight * df * dh * GetModel(angle, h_here, f_here, t, s_norm, out='kapton')
	
	path_length /= 4 * f_range * h_range
	return MOD.GetAbsorption(path_length, kapton_absorption_length)


mpl.rc_file('matplotlibrc.txt')
angle = 0
h = 0.045
f = 1.29
t = 0.025

data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=0.9)
"""
# Test two axis integration
absorption_1 = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=1)
absorption_f = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=11)
absorption_h = IntegrateModel2(angle, h, f, t, s_norm, kapton_absorption_length, 0.0001, 0.01, n=11)
absorption_fh = IntegrateModel2(angle, h, f, t, s_norm, kapton_absorption_length, 0.2, 0.01, n=11)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
v_lims = [0.9, 1.0]
im0 = axes[0].imshow(absorption_1, vmin=v_lims[0], vmax=v_lims[1])
im1 = axes[1].imshow(absorption_f, vmin=v_lims[0], vmax=v_lims[1])
im2 = axes[2].imshow(absorption_h, vmin=v_lims[0], vmax=v_lims[1])
im3 = axes[3].imshow(absorption_fh, vmin=v_lims[0], vmax=v_lims[1])
for index in range(3):
	axes[index].set_xticks([])
	axes[index].set_yticks([])

fig, axes = plt.subplots(1, 1)
axes.plot(absorption_1.mean(axis=0), label='None')
axes.plot(absorption_f.mean(axis=0), label='f')
axes.plot(absorption_h.mean(axis=0), label='h')
axes.plot(absorption_fh.mean(axis=0), label='fh')
axes.legend()
plt.show()

"""
# Test water
absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
absorption_water = WaterAbsorption(angle, h, f, t, s_norm, v=-0.05)
absorption_total = absorption * absorption_water
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
v_lims = [0.9, 1.0]
im0 = axes[0].imshow(normalized_image, vmin=0.9, vmax=1.1)
im1 = axes[1].imshow(absorption, vmin=v_lims[0], vmax=v_lims[1])
im2 = axes[2].imshow(absorption_water, vmin=v_lims[0], vmax=v_lims[1])
im3 = axes[3].imshow(absorption_total, vmin=v_lims[0], vmax=v_lims[1])
for index in range(4):
	axes[index].set_xticks([])
	axes[index].set_yticks([])

fig, axes = plt.subplots(1, 1)
axes.plot(normalized_image.mean(axis=0), label='Image')
axes.plot(absorption.mean(axis=0), label='Kapton')
axes.plot(absorption_water.mean(axis=0), label='Water')
axes.plot(absorption_total.mean(axis=0), label='Total')
axes.legend()
plt.show()

"""
# Test different n's
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
for n in [1, 3, 5, 11]:
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=n)
	axes[0].plot(absorption.mean(axis=0), label=str(n))
	axes[1].plot(absorption.mean(axis=0), label=str(n))
axes[0].set_ylabel('Absorption Model')
axes[0].set_xlabel('Pixels along fast axis')
axes[1].set_xlabel('Pixels along fast axis')
axes[0].set_title('Absorption averaged  along slow axis')
axes[1].set_title('Zoomed to maximum absorption')
axes[1].set_xlim([870, 910])
axes[1].set_ylim([0.85, 0.89])
axes[1].legend(title='N')
plt.tight_layout()
plt.show()
"""