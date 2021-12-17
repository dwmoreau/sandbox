import sys
sys.path.append('/home/david/dials_dev/modules/dxtbx/src')
sys.path.append('/home/david/dials_dev/build/lib')
sys.path.append('/home/david/dials_dev/modules')
from dxtbx.model.experiment_list import ExperimentListFactory
import matplotlib.pyplot as plt
import numpy as np


pixel_size = 0.1
detector_shape = [1000, 1000]
detector_distance = 200
beam_center = [500, 500]

h = 0.05
f = 1
t = 0.05
angle = 10 * np.pi/180
r_drop = 0.2

# https://henke.lbl.gov/optical_constants/atten2.html
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# absorption at 10 keV
kapton_absorption_length = 2.34804
water_absorption_coefficient = 5.329 #cm^2 / g
water_density = 1 # g / cm^3
water_absorption_length = 1 / (water_density * water_absorption_coefficient)

################
# using angles #
################
xx, yy = np.meshgrid(
	np.arange(detector_shape[0]) - beam_center[0],
	np.arange(detector_shape[1]) - beam_center[1],
	)

theta2 = np.arctan(pixel_size * np.sqrt(xx**2 + yy**2) / detector_distance)
phi = np.arctan2(yy, xx) + angle
metric = -np.cos(phi) * np.tan(theta2)

indices1 = np.logical_and(
	metric >= h / f,
	metric < (h + t) / f
	)
indices2 = metric >= (h + t) / f

path_length = np.zeros(detector_shape)
path_length[indices1] = f / np.cos(theta2[indices1]) + h / (np.sin(theta2[indices1]) * np.cos(phi[indices1]))
path_length[indices2] = -t / (np.sin(theta2[indices2]) * np.cos(phi[indices2]))

correction = np.exp(path_length / kapton_absorption_length)

"""
R = pixel_size * np.sqrt(xx**2 + yy**2)
fig, axes = plt.subplots(1, 5, figsize=(10, 6))
axes[0].imshow(xx, origin='lower')
axes[1].imshow(yy, origin='lower')
axes[2].imshow(R, origin='lower')
axes[3].imshow(theta2 * 180/np.pi, origin='lower')
axes[4].imshow(phi * 180/np.pi, origin='lower')
plt.show()

regions = np.zeros(detector_shape)
regions[indices1] = 1
regions[indices2] = 2

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(metric, origin='lower')
axes[1].imshow(regions, origin='lower')
#plt.show()

absorption = np.exp(-path_length / kapton_absorption_length)
fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes[0].imshow(path_length, origin='lower')
axes[1].imshow(absorption, origin='lower')
axes[2].imshow(correction, origin='lower')
#plt.show()
"""

#################################
# Using vectors and projections #
#################################
xx, yy = np.meshgrid(
	np.arange(detector_shape[0]) - beam_center[0],
	np.arange(detector_shape[1]) - beam_center[1],
	)
zz = detector_distance * np.ones(detector_shape)
s = np.stack((pixel_size * yy, pixel_size * xx, zz), axis=2)
s_norm = s.copy()
for index in range(3):
	s_norm[:, :, index] = s[:, :, index] / np.linalg.norm(s, axis=2)

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

path_length = np.zeros(detector_shape)
path_length[indices2] = L3[indices2] - L1[indices2]
path_length[indices3] = L2[indices3] - L1[indices3]
correction = np.exp(path_length / kapton_absorption_length)

n1_d_snorm = np.matmul(s_norm, n1)
L4 = -n1_d_snorm + np.sqrt(n1_d_snorm**2 - h**2 + r_drop**2)
indices_water1 = np.logical_and(
	L1 < L4,
	L1 >= 0
	)
path_length_water = np.zeros(detector_shape)
path_length_water[np.invert(indices_water1)] = L4[np.invert(indices_water1)]
path_length_water[indices_water1] = L1[indices_water1]
water_correction = np.exp(path_length_water / water_absorption_length)

fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes[0].imshow(correction, origin='lower')
axes[1].imshow(water_correction, origin='lower')
axes[2].imshow(correction*water_correction, origin='lower')
#plt.show()

"""
regions = np.zeros(detector_shape)
regions[indices2] = 1
regions[indices3] = 2
fig, axes = plt.subplots(1, 5, figsize=(10, 6))
axes[0].imshow(L1, origin='lower', vmin=0, vmax=1)
axes[1].imshow(L2, origin='lower', vmin=0, vmax=1)
axes[2].imshow(L3, origin='lower', vmin=0, vmax=1)
axes[3].imshow(L4, origin='lower', vmin=0, vmax=1)
axes[4].imshow(regions, origin='lower')
#plt.show()
"""
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
axes[0].imshow(path_length, origin='lower')
axes[1].imshow(path_length_water, origin='lower')
plt.show()
