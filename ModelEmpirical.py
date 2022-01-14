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


def Fit(params, t, normalized_image, s_norm, kapton_absorption_length, mask):
	angle = params[0]
	h = params[1]
	f = params[2]
	absorption = MOD.IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, 0.1, n=3)
	residuals = (normalized_image - absorption)[np.invert(mask)]
	return np.linalg.norm(residuals)

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
t = 0.025

data, normalized_image, s, s_norm, mask, kapton_absorption_length = MOD.GetImage(polarization_fraction=0.90)
normalized_image[mask] = np.nan
Y = MOD.Gradient(s, normalized_image, mask)
normalized_image = normalized_image / Y

absorption = normalized_image.mean(axis)