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


image_file_name = '/home/david/Documents/Background/Data/converted.cbf'
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

for index in range(len(data)):
	panel = data[index].as_numpy_array()
	fig, axes = plt.subplots(1, 1)
	axes.imshow(panel)
	plt.show()