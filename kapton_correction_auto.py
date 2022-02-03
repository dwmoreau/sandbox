"""
sys.path.append('/net/dials/raid1/dwmoreau/dials_dev2/modules/dxtbx/src')
sys.path.append('/net/dials/raid1/dwmoreau/dials_dev2/build/lib')
sys.path.append('/net/dials/raid1/dwmoreau/dials_dev2/modules')
ToDo
    - Large scale testing
        x parallize by frame
        x run on dials
        - average first 60 frames
            x store peak intensity from the intensity histogram for weighting
            x average based on weights and masks
            - determine average model
            - apply this constant model
            - beamer presentation
        - Rayonix frame

    - Read Ke 2018
    
    Action items - 2/1/2022
        x Convert to photons
        x profile and speed up
        x target function: model - data
            - Normalized image as model
            - divide raw image by normalized image to get "Data" 
            - divide raw image by model to get "Model"
            - difference between model and data
        - Z-score
            - get standard error for each pixel
        - compare vs constant model
            - determine based on average image from the first 60 frames
                - image average
            - does the frame by frame model improve over the single model approach?

        - Speed up:
            - order data so it can be sliced instead of indexed
            - GPU on dials???

        - get working on the rayonix single panel image

        - 45 degree angle
            - apply model at 45 degrees to a frame and use that as a test

    uncertainty
        - Read 
            - Evans 11 - EV11
            - Brewster 2019
            - normal probability plots
    libtbx.python `libtbx.find_in_repositories dxtbx`/src/dxtbx/format/cbf_writer.py filename 

    libtbx.find_in_repositories dxtbx/src/dxtbx/format/cbf_writer.py /net/dials/raid1/sauter/bernina/spectrum_masters/run_000795.JF07T32V01_master.h5 0
"""
import sys
sys.path.append('/home/david/dials_dev/modules/dxtbx/src')
sys.path.append('/home/david/dials_dev/build/lib')
sys.path.append('/home/david/dials_dev/modules')
from cctbx import factor_kev_angstrom
from dials.algorithms.integration.kapton_correction import get_absorption_correction
from dials.array_family import flex
import dxtbx
import json
import matplotlib.pyplot as plt
import numpy as np
import numba
import os
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.signal import medfilt


# https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
# This is faster by a factor of 5 - 10
@numba.jit(nopython=True)
def numba_histogram(a, bins):
    n = bins.size - 1
    a_min = bins[0]
    a_max = bins[-1]
    factor = n / (a_max - a_min)
    hist = np.zeros(n, dtype=np.intp)
    for x in a:
        bin = int(factor * (x - a_min))
        if bin >=0 and bin < n:
            hist[int(bin)] += 1
    return hist


@numba.jit(nopython=True)
def numba_histogram_weighted(a, bins, weights):
    n = bins.size - 1
    a_min = bins[0]
    a_max = bins[-1]
    factor = n / (a_max - a_min)
    hist = np.zeros(n, dtype=np.float64)
    for index, x in enumerate(a):
        bin = int(factor * (x - a_min))
        if bin >=0 and bin < n:
            hist[int(bin)] += weights[index]
    return hist


@numba.jit(nopython=True)
def numba_histogram2d(x, y, bins):
    bins_x = bins[0]
    bins_y = bins[1]
    n = x.size
    nx = bins_x.size - 1
    ny = bins_y.size - 1
    min_x = bins_x[0]
    max_x = bins_x[-1]
    min_y = bins_y[0]
    max_y = bins_y[-1]
    factor_x = nx / (max_x - min_x)
    factor_y = ny / (max_y - min_y)
    hist = np.zeros((nx, ny), dtype=np.intp)
    for index in range(n):
        binx = int(factor_x * (x[index] - min_x))
        biny = int(factor_y * (y[index] - min_y))
        if binx >=0 and binx < nx and biny >=0 and biny < ny:
            hist[binx, biny] += 1
    return hist


@numba.jit(nopython=True)
def numba_histogram2d_weighted(x, y, bins, weights):
    bins_x = bins[0]
    bins_y = bins[1]
    n = x.size
    nx = bins_x.size - 1
    ny = bins_y.size - 1
    min_x = bins_x[0]
    max_x = bins_x[-1]
    min_y = bins_y[0]
    max_y = bins_y[-1]
    factor_x = nx / (max_x - min_x)
    factor_y = ny / (max_y - min_y)
    hist = np.zeros((nx, ny), dtype=np.intp)
    for index in range(n):
        binx = int(factor_x * (x[index] - min_x))
        biny = int(factor_y * (y[index] - min_y))
        if binx >=0 and binx < nx and biny >=0 and biny < ny:
            hist[binx, biny] += weights[index]
    return hist


@numba.jit(nopython=True)
def numba_binned_mean(x, y, bins, weights):
    bins_x = bins[0]
    bins_y = bins[1]
    n = x.size
    nx = bins_x.size - 1
    ny = bins_y.size - 1
    min_x = bins_x[0]
    max_x = bins_x[-1]
    min_y = bins_y[0]
    max_y = bins_y[-1]
    factor_x = nx / (max_x - min_x)
    factor_y = ny / (max_y - min_y)
    counts = np.zeros((nx, ny), dtype=np.intp)
    summed = np.zeros((nx, ny), dtype=np.float64)
    averaged = np.zeros((nx, ny), dtype=np.float64)
    for index in range(n):
        binx = int(factor_x * (x[index] - min_x))
        biny = int(factor_y * (y[index] - min_y))
        if binx >=0 and binx < nx and biny >=0 and biny < ny:
            counts[binx, biny] += 1
            summed[binx, biny] += weights[index]
    averaged = summed / counts
    return averaged, counts


class kapton_correction_auto():
    def __init__(self, file_name, params, results_dir, frame=None):
        self.save_to_dir = results_dir
        self.file_name = file_name
        self.frame = frame

        # parameters associated with the:
        #   kapton absorption model
        self.angle = params['angle']
        self.h = params['h']
        self.f = params['f']
        self.t = params['t']

        #   finite pathlength through the sample
        self.f_range = params['f_range']
        self.n_int = params['n_int']

        # parameters associated with the algorithm
        #   pad - distance from edge of detector panels to mask
        #   max_intensity - if a pixel is larger than this, include in the mask
        #       max_intensity_limit - initial guess - should just be very large
        #       max_intensity - determined based on histogram
        #   bad_panels - panels to exclude from analysis
        self.pad = params['pad']
        self.max_intensity_limit = params['max_intensity_limit']
        self.max_intensity = params['max_intensity_limit']
        self.bad_panels = params['bad_panels']

        # Fraction of polarized light
        self.polarization_fraction = params['polarization_fraction']

        # Parameter associated with initial angle determination
        self.max_abs_point = params['max_abs_point']
        return None

    def load_previous(self):
        if self.frame is not None:
            self.save_to_dir = self.save_to_dir + '/Frame_' + str(self.frame).zfill(4)
        with open(self.save_to_dir + '/Params_' + str(self.frame).zfill(4) + '.json', 'r') as json_file:
            params_json = json.load(json_file)
        self.angle = params_json['angle']
        self.h = params_json['h']
        self.f = params_json['f']
        self.t = params_json['t']
        self.f_range = params_json['f_range']
        self.n_int = params_json['n_int']
        self.pad = params_json['pad']
        self.polarization_fraction = params_json['polarization_fraction']
        self.max_intensity_limit = params_json['max_intensity_limit']
        self.max_intensity = params_json['max_intensity']
        self.wavelength = params_json['wavelength']
        self.n_panels = params_json['n_panels']
        self.pixel_size = params_json['pixel_size']
        self.panel_shape = params_json['panel_shape']
        self.panel_size = params_json['panel_size']
        self.kapton_absorption_length = params_json['kapton_absorption_length']
        self.n_pixels_full = params_json['n_pixels_full']
        self.save_to_dir = params_json['save_to_dir']
        self.file_name = params_json['file_name']
        self.bad_panels = params_json['bad_panels']

        flattened = np.load(
            self.save_to_dir + '/Flattened_Frame_' + str(self.frame).zfill(4) + '.npy'
            )
        arrayed = np.load(
            self.save_to_dir + '/Arrayed_Frame_' + str(self.frame).zfill(4) + '.npy'
            )
        averaged = np.load(
            self.save_to_dir + '/Averaged_Frame_' + str(self.frame).zfill(4) + '.npy'
            )

        self.I = flattened[:, 0]
        self.s = np.zeros((self.I.size, 3))
        self.s_norm = np.zeros((self.I.size, 3))
        self.s[:, 0] = flattened[:, 1]
        self.s[:, 1] = flattened[:, 2]
        self.s[:, 2] = flattened[:, 3]
        self.s_norm[:, 0] = flattened[:, 4]
        self.s_norm[:, 1] = flattened[:, 5]
        self.s_norm[:, 2] = flattened[:, 6]
        self.theta2 = flattened[:, 7]
        self.phi = flattened[:, 8]
        self.polarization = flattened[:, 9]
        self.integrated_image = flattened[:, 10]
        self.normalized_image = self.I / (self.polarization * self.integrated_image)

        self.I_array = arrayed[:, :, 0]
        self.s_array = np.zeros((*self.I_array.shape, 3))
        self.s_norm_array = np.zeros((*self.I_array.shape, 3))
        self.s_array[:, :, 0] = arrayed[:, :, 1]
        self.s_array[:, :, 1] = arrayed[:, :, 2]
        self.s_array[:, :, 2] = arrayed[:, :, 3]
        self.s_norm_array[:, :, 0] = arrayed[:, :, 4]
        self.s_norm_array[:, :, 1] = arrayed[:, :, 5]
        self.s_norm_array[:, :, 2] = arrayed[:, :, 6]
        self.mask_array = arrayed[:, :, 7].astype(np.bool)
        self.theta2_array = arrayed[:, :, 8]
        self.phi_array = arrayed[:, :, 9]
        self.polarization_array = arrayed[:, :, 10]
        self.integrated_image_array = arrayed[:, :, 11]
        self.normalized_image_array =\
            self.I_array / (self.polarization_array * self.integrated_image_array)

        self.az_average = averaged[:, :, 0]
        self.az_average_array = averaged[:, :, 1]
        return None

    def load_file(self):
        self.image = dxtbx.load(self.file_name)
        self.detector = self.image.get_detector()
        self.beam = self.image.get_beam()
        self.wavelength = self.beam.get_wavelength()
        self.energy = factor_kev_angstrom / self.wavelength
        return None

    def save_params(self):
        # Save parameters & fit params
        params = {
            'angle': self.angle,
            'h': self.h,
            'f': self.f,
            't': self.t,
            'f_range': self.f_range,
            'n_int': self.n_int,
            'pad': self.pad,
            'polarization_fraction': self.polarization_fraction,
            'max_intensity_limit': self.max_intensity_limit,
            'max_intensity': self.max_intensity,
            'wavelength': self.wavelength,
            'n_panels': self.n_panels,
            'pixel_size': self.pixel_size,
            'panel_shape': self.panel_shape,
            'panel_size': self.panel_size,
            'kapton_absorption_length': self.kapton_absorption_length,
            'n_pixels_full': self.n_pixels_full,
            'save_to_dir': self.save_to_dir,
            'file_name': self.file_name,
            'bad_panels': self.bad_panels,
            }

        with open(self.save_to_dir + '/Params_' + str(self.frame).zfill(4) + '.json', 'w') as file:
            file.write(json.dumps(params))
        return None

    def import_and_save(self):
        if self.frame is not None:
            self.save_to_dir = self.save_to_dir + '/Frame_' + str(self.frame).zfill(4)
        if os.path.exists(self.save_to_dir) is False:
            os.mkdir(self.save_to_dir)
        self.load_file()        
        self.n_panels = len(self.detector)
        self.pixel_size = self.detector[0].get_pixel_size()[0]
        self.panel_shape = self.detector[0].get_image_size()
        self.panel_size = self.panel_shape[0] * self.panel_shape[1]
        self.kapton_absorption_length\
            = get_absorption_correction()(self.wavelength)
        self.n_pixels_full = self.n_panels * self.panel_size

        self.get_frame()
        self.get_image_array()
        self.get_max_intensity()
        self.get_image()
        self.save_params()

        # Save important arrays
        flattened = np.column_stack((
            self.I,
            self.s[:, 0],
            self.s[:, 1],
            self.s[:, 2],
            self.s_norm[:, 0],
            self.s_norm[:, 1],
            self.s_norm[:, 2],
            self.theta2,
            self.phi,
            self.polarization,
            self.integrated_image
            ))
        
        arrayed = np.stack((
            self.I_array,
            self.s_array[:, :, 0],
            self.s_array[:, :, 1],
            self.s_array[:, :, 2],
            self.s_norm_array[:, :, 0],
            self.s_norm_array[:, :, 1],
            self.s_norm_array[:, :, 2],
            self.mask_array,
            self.theta2_array,
            self.phi_array,
            self.polarization_array,
            self.integrated_image_array
            ), axis=-1)

        averaged = np.stack((
            self.az_average,
            self.az_average_array
            ), axis=-1)

        np.save(
            self.save_to_dir + '/Flattened_Frame_' + str(self.frame).zfill(4) + '.npy',
            flattened
            )
        np.save(
            self.save_to_dir + '/Arrayed_Frame_' + str(self.frame).zfill(4) + '.npy',
            arrayed
            )
        np.save(
            self.save_to_dir + '/Averaged_Frame_' + str(self.frame).zfill(4) + '.npy',
            averaged
            )
        return None

    def get_frame(self):
        if self.frame is None:
            self.data = self.image.get_raw_data()
        else:
            self.data = self.image.get_raw_data(self.frame)
        return None

    def _get_polarization(self, phi, theta2):
        factor = np.cos(2*phi)*np.sin(theta2)**2 / (1+np.cos(theta2)**2)
        return 1 - self.polarization_fraction * factor

    def _azimuthal_average(self, theta2, I, phi, mask=None):
        # Returns the azimuthally averaged image
        # The average is calculated with histograms - very fast implementation
        #   counts == number of pixels in a bin
        #   sum == summed intensity in a bin
        #   sum / counts == average pixel intensity
        theta2_bins = np.linspace(0, 60, 61) * np.pi/180
        theta2_centers = (theta2_bins[1:] + theta2_bins[:-1]) / 2
        phi_indices = np.logical_or(
            phi < np.pi/2,
            phi > np.pi/180 * (270 + 10)
            )
        if mask is None:
            integration_sum = numba_histogram_weighted(
                theta2[phi_indices], bins=theta2_bins, weights=I[phi_indices]
                )
            integration_counts = numba_histogram(theta2[phi_indices], bins=theta2_bins)
        else:
            mask_inv = np.logical_and(
                np.invert(mask),
                phi_indices
                )
            integration_sum = numba_histogram_weighted(
                theta2[mask_inv],
                bins=theta2_bins,
                weights=I[mask_inv]
                )
            integration_counts = numba_histogram(
                theta2[mask_inv],
                bins=theta2_bins
                )
        integrated = integration_sum / integration_counts
        az_average = np.column_stack((theta2_centers, integrated))
        indices = np.invert(np.isnan(integrated))
        integrated_image = np.interp(
            theta2, theta2_centers[indices], integrated[indices]
            )
        return integrated_image, az_average

    def _get_theta2_phi(self, array=False):
        # If working with 2D arrays
        if array:
            s0 = self.s_array[:, :, 0]
            s1 = self.s_array[:, :, 1]
            s2 = self.s_array[:, :, 2]
        # If working with flattened arrays
        else:
            s0 = self.s[:, 0]
            s1 = self.s[:, 1]
            s2 = self.s[:, 2]
        R = np.sqrt(s0**2 + s1**2)
        theta2 = np.arctan(R / np.abs(s2))
        phi = np.pi + np.arctan2(s1, -1*s0)
        return theta2, phi

    def _process_panel(self, data, panel):
        # This returns information about an individual panel to be fed into
        # a single array to represent the entire image

        # s - vector from crystal to each pixel
        x, y = np.meshgrid(
            np.arange(self.panel_shape[0]),
            np.arange(self.panel_shape[1])
            )
        mm = panel.pixel_to_millimeter(flex.vec2_double(
            flex.double(x.ravel()),
            flex.double(y.ravel())
            ))
        s = panel.get_lab_coord(mm).as_numpy_array()

        I = data.as_numpy_array().ravel() / self.energy
        s_norm = (s.T / np.linalg.norm(s, axis=1)).T
        s_norm[:, 2] *= -1

        mask = np.zeros(self.panel_shape, dtype=np.bool)
        # Mask out the edges of the panel
        mask[:self.pad, :] = True
        mask[-self.pad:, :] = True
        mask[:, :self.pad] = True
        mask[:, -self.pad:] = True
        mask = mask.ravel()

        # Mask pixels with intensities less than zero under the assumption
        # these are bad pixels
        mask = np.logical_or(
            mask,
            I <= 0
            )
        # Masks out the pixels with large intensities assuming they contain
        # bragg diffraction
        mask = np.logical_or(
            mask,
            I >= self.max_intensity
            )
        return I, s, s_norm, mask

    def get_image(self):
        # Turns the file into single arrays that are used for the analysis
        self.I = np.zeros(self.n_pixels_full)
        self.s = np.zeros((self.n_pixels_full, 3))
        self.s_norm = np.zeros((self.n_pixels_full, 3))
        self.mask = np.zeros(self.n_pixels_full, dtype=np.bool)
        for index in range(self.n_panels):
            if index in self.bad_panels:
                self.mask[start: end] = True
            else:
                I_panel, s_panel, s_norm_panel, mask_panel, \
                    = self._process_panel(self.data[index], self.detector[index])
                start = index * self.panel_size
                end = (index + 1) * self.panel_size
                self.I[start: end] = I_panel
                self.s[start: end] = s_panel
                self.s_norm[start: end] = s_norm_panel
                self.mask[start: end] = mask_panel
        self.mask = np.logical_or(
            self.mask,
            self.I <= 0
            )
        self.n_pixels = np.invert(self.mask).sum()
        self.I = np.delete(self.I, self.mask)
        self.s = np.delete(self.s, self.mask, axis=0)
        self.s_norm = np.delete(self.s_norm, self.mask, axis=0)
        self.theta2, self.phi = self._get_theta2_phi()
        self.polarization = self._get_polarization(self.phi, self.theta2)
        self.integrated_image, self.az_average = self._azimuthal_average(
            self.theta2,
            (self.I / self.polarization),
            self.phi
            )
        self.normalized_image\
            = self.I / (self.polarization * self.integrated_image)
        return None

    def get_image_array(self):
        # This sets up the image as a 2D array. This is used entirely for
        # visualization purposes.

        # Step 1: create arrays big enough to fit the data
        #   loop through all the panels and find the minimimum and maximum
        #   panel positions. This gives the array size
        # Step 2: populate the arrays with the data

        min_pos = np.zeros(2)
        max_pos = np.zeros(2)
        for index in range(self.n_panels):
            origin = self.detector[index].get_origin()
            if origin[0] < min_pos[0]:
                min_pos[0] = origin[0]
            if origin[1] < min_pos[1]:
                min_pos[1] = origin[1]
            if origin[0] > max_pos[0]:
                max_pos[0] = origin[0]
            if origin[1] > max_pos[1]:
                max_pos[1] = origin[1]
        max_pos = np.round(
            (max_pos - min_pos) / self.pixel_size,
            decimals=0
            ).astype(np.int)
        max_pos += self.panel_shape
        self.I_array = -1*np.ones(max_pos[::-1])
        self.s_array = -1*np.ones((*max_pos[::-1], 3)).ravel()
        self.s_norm_array = -1*np.ones((*max_pos[::-1], 3)).ravel()
        self.mask_array = np.zeros(max_pos[::-1], dtype=np.bool)

        for index in range(self.n_panels):
            origin = np.round(
                (self.detector[index].get_origin()[:2] - min_pos) / self.pixel_size,
                decimals=0
                ).astype(np.int)
            I_panel, s_panel, s_norm_panel, mask_panel = self._process_panel(
                self.data[index], self.detector[index]
                )
            indices = np.zeros(max_pos[::-1], dtype=np.bool)
            indices[
                origin[1]: origin[1] + self.panel_shape[1],
                origin[0]: origin[0] + self.panel_shape[0]
                ] = True
            indices_flat = np.zeros((*max_pos[::-1], 3), dtype=np.bool)
            indices_flat[
                origin[1]: origin[1] + self.panel_shape[1],
                origin[0]: origin[0] + self.panel_shape[0],
                :
                ] = True
            indices_flat = indices_flat.ravel() 
            self.I_array[indices] = I_panel
            self.mask_array[indices] = mask_panel
            self.s_array[indices_flat] = s_panel.ravel() 
            self.s_norm_array[indices_flat] = s_norm_panel.ravel()
            if index in self.bad_panels:
                self.mask_array[indices] = True
        self.s_array = self.s_array.reshape((*max_pos[::-1], 3))
        self.s_norm_array = self.s_norm_array.reshape((*max_pos[::-1], 3))
        self.mask_array = np.logical_or(
            self.mask_array,
            self.I_array <= 0
            )
        self.theta2_array, self.phi_array = self._get_theta2_phi(array=True)
        self.theta2_array[self.mask_array] = -1
        self.phi_array[self.mask_array] = -1
        self.polarization_array = self._get_polarization(
            self.phi_array, self.theta2_array
            )
        self.integrated_image_array, self.az_average_array = self._azimuthal_average(
            self.theta2_array.ravel(),
            (self.I_array / self.polarization_array).ravel(),
            self.phi_array.ravel(),
            mask=self.mask_array.ravel()
            )
        self.integrated_image_array = self.integrated_image_array.reshape(max_pos[::-1])
        self.normalized_image_array\
            = self.I_array / (self.polarization_array * self.integrated_image_array)
        return None

    def get_path_length(self,  params=None, array=False):
        def distance(n, s_norm):
            return np.divide(np.linalg.norm(n)**2, np.matmul(s_norm, n))

        if params is None:
            angle = self.angle
            h = self.h
            f = self.f
        else:
            angle = params[0]
            h = params[1]
            f = params[2]

        if array:
            s_norm = self.s_norm_array
        else:
            s_norm = self.s_norm

        # Returns the pathlength through the kapton film

        # s_norm: unit vector pointing from crystal to each detector pixel
        # n1: Vector pointing from crystal to front face of kapton
        # n2: Vector pointing from crystal to back face of kapton
        # n3: Vector pointing from crystal to far edge of kapton
        n1 = -h * np.array((1, 0, 0)).T
        n2 = -(h + self.t) * np.array((1, 0, 0)).T
        n3 = f * np.array((0, 0, 1)).T

        # Rotate these vectors to account for the kapton angle
        Rz = np.array((
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1)
            ))
        n1 = np.matmul(Rz, n1)
        n2 = np.matmul(Rz, n2)
        n3 = np.matmul(Rz, n3)

        # Calculate the distance from the crystal to each kapton
        # face along each unit vector pointing to the pixels
        # These assume that the kapton faces are infinite planes
        
        # These are slow - bottleneck
        L1 = distance(n1, s_norm)
        L2 = distance(n2, s_norm)
        L3 = distance(n3, s_norm)
        
        # indices1: Paths that pass through the front face and the far edge
        # indices2: Paths that pass through the front face and the back face
        indices1 = np.logical_and(L1 < L3, L2 >= L3)
        indices2 = np.logical_and(L3 > L2, L2 >= 0)

        path_length = np.zeros(s_norm.shape[:-1])
        # For paths through the front face and the far edge
        path_length[indices1] = L3[indices1] - L1[indices1]
        # For paths through the front face and the back face
        path_length[indices2] = L2[indices2] - L1[indices2]

        if self.n_int != 1:
            # This calculates the model under the assumption there is a finite
            # pathlength through the solvent. It integrates the kapton pathlength
            # through the solvent pathlength.
            # This is equivalent to a weighted average of the kapton pathlength
            # for different values of f where the end cases have half the weight
            # as the rest of the f values.
            path_length_int = np.zeros((*s_norm.shape[:-1], self.n_int))
            f_int = np.linspace(f - self.f_range/2, f + self.f_range/2, self.n_int)
            for index, f_here in enumerate(f_int):
                n3 = f_here * np.array((0, 0, 1)).T
                n3 = np.matmul(Rz, n3)
                L3 = distance(n3, s_norm)
                indices1 = np.logical_and(
                    L1 < L3,
                    L2 >= L3
                    )
                indices2 = np.logical_and(
                    L3 > L2,
                    L2 >= 0
                    )
                path_length_int[indices1, index] = L3[indices1] - L1[indices1]
                path_length_int[indices2, index] = L2[indices2] - L1[indices2]
            path_length = np.trapz(path_length_int, f_int, axis=-1) / self.f_range

        if array:
            return path_length.reshape(self.I_array.shape)
        else:
            return path_length

    def _absorption_calc(self, L, L_abs):
        return np.exp(-L / L_abs)

    def get_absorption(self, array=False):
        if array:
            self.path_length_array = self.get_path_length(array=True)
            self.absorption_array = self._absorption_calc(
                self.path_length_array, self.kapton_absorption_length
                )
        else:
            self.path_length = self.get_path_length()
            self.absorption = self._absorption_calc(
                self.path_length, self.kapton_absorption_length
                )
        return None

    def get_absorption_opt(self, s_norm_opt, phi_indices, angle, h, f):
        def distance(n, s_norm):
            return np.linalg.norm(n)**2 / np.matmul(s_norm, n)
        
        n1 = -h * np.array((1, 0, 0)).T
        n2 = -(h + self.t) * np.array((1, 0, 0)).T
        n3 = f * np.array((0, 0, 1)).T

        Rz = np.array((
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1)
            ))
        n1 = np.matmul(Rz, n1)
        n2 = np.matmul(Rz, n2)
        n3 = np.matmul(Rz, n3)

        L1 = distance(n1, s_norm_opt)
        L2 = distance(n2, s_norm_opt)
        L3 = np.linalg.norm(n3)**2 / (s_norm_opt[:, 2] * n3[2])
        
        L2_L3 = L2 >= L3
        indices1 = np.logical_and(L1 < L3, L2_L3)
        indices2 = np.logical_and(np.invert(L2_L3), L2 >= 0)

        path_length = np.zeros(s_norm_opt.shape[:-1])
        path_length[indices1] = L3[indices1] - L1[indices1]
        np.place(path_length, indices2, np.extract(indices2, L2-L1))
        
        absorption = np.ones(self.s_norm.shape[:-1])
        absorption[phi_indices] = np.exp(-path_length / self.kapton_absorption_length)
        return absorption

    def check_polarization(self):
        # This averages the normalized images radially at constant phi angles
        # where the normalized images are computed for different polarization
        # fractions. The plot should be constant for the correct polarization
        # fraction.
        original_p_frac = self.polarization_fraction
        phi_bins = np.linspace(0, 2*np.pi, 64)
        phi_centers = (phi_bins[1:] + phi_bins[:-1]) / 2
        indices = np.logical_and(
            self.theta2 >= 10 * np.pi/180,
            self.theta2 <= 40 * np.pi/180
            )
        counts = numba_histogram(self.phi[indices], bins=phi_bins)
        p_frac = [0, 0.8, 0.85, 0.9, 0.95, 1.0]
        integrated = np.zeros((len(p_frac), phi_centers.size))
        fig, axes = plt.subplots(1, 1)
        for index, p in enumerate(p_frac):
            self.polarization_fraction = p
            polarization = self._get_polarization(self.phi, self.theta2)
            normalized_image = self.I / (polarization * self.integrated_image)
            phi_sum = numba_histogram_weighted(
                self.phi[indices],
                bins=phi_bins,
                weights=normalized_image[indices]
                )
            integrated[index, :] = phi_sum / counts
            axes.plot(
                180/np.pi * phi_centers, integrated[index, :],
                label='%0.2f' % (p)
                )
        axes.legend(title='Polarization\nFraction')
        axes.set_xlabel('Phi (degrees)')
        axes.set_ylabel('Intensity')
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/PolarizationCheck.png')
        plt.close(fig)
        self.polarization_fraction = original_p_frac
        return None

    def plot_results(self):
        I_corrected = self.I / self.polarization / self.absorption
        self.IDist(I_corrected, plot=True)
        self.gradients(I_corrected, plot=True)

        # x Histogram of raw image and normalized image
        bins_raw_image = np.arange(0, self.max_intensity, 0.1)
        hist_raw_image, b = np.histogram(
            self.I_array[np.invert(self.mask_array)],
            bins=bins_raw_image,
            density=True
            )
        centers_raw_image = (bins_raw_image[1:] + bins_raw_image[:-1]) / 2
        raw_image_width = bins_raw_image[1] - bins_raw_image[0]

        bins_norm_image = np.linspace(0, 2, 101)
        centers_norm_image = (bins_norm_image[1:] + bins_norm_image[:-1]) / 2
        norm_image_width = bins_norm_image[1] - bins_norm_image[0]

        hist_norm_image, b = np.histogram(
            self.normalized_image_array[np.invert(self.mask_array)],
            bins=bins_norm_image,
            density=True
            )
        hist_norm_image_corrected, b = np.histogram(
            (self.normalized_image_array / self.absorption_array)[np.invert(self.mask_array)],
            bins=bins_norm_image,
            density=True
            )

        fig, axes = plt.subplots(1, 3, figsize=(10, 6))
        axes[0].bar(centers_raw_image, hist_raw_image, raw_image_width)
        axes[1].bar(centers_norm_image, hist_norm_image, norm_image_width)
        axes[2].bar(centers_norm_image, hist_norm_image_corrected, norm_image_width)
        ylim = axes[0].get_ylim()
        axes[1].plot([1, 1], ylim, color=[0, 0, 0], linestyle=':')
        axes[2].plot([1, 1], ylim, color=[0, 0, 0], linestyle=':')
        axes[0].set_ylim(ylim)
        axes[0].set_title('Raw Image')
        axes[1].set_title('Normalized Image')
        axes[2].set_title('Normalized & Corrected Image')
        axes[0].set_xlabel('Intensity')
        axes[1].set_xlabel('Intensity')
        axes[2].set_xlabel('Intensity')
        axes[0].set_ylabel('Number of Pixels')
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/Intensity_Histograms.png')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(8, 6))
        im = [[] for index in range(2)]
        im[0] = axes[0].imshow(self.absorption_array, vmin=0.8, vmax=1.2)
        im[1] = axes[1].imshow(self.normalized_image_array, vmin=0.8, vmax=1.2)
        for index in range(2):
            axes[index].set_xticks([])
            axes[index].set_yticks([])
        fig.colorbar(im[1], ax=axes[1])
        axes[0].set_title('Kapton Absorption')
        axes[1].set_title('Normalized Image')
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/Absorption.png')
        plt.close(fig)

        # theta2 & phi
        theta2 = 180/np.pi * self.theta2_array.copy()
        theta2[self.mask_array] = np.nan
        phi = 180/np.pi * self.phi_array.copy()
        phi[self.mask_array] = np.nan
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        im0 = axes[0].imshow(theta2)
        im1 = axes[1].imshow(phi)
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        for index in range(2):
            axes[index].set_xticks([])
            axes[index].set_yticks([])
        axes[0].set_title('2$\\theta$')
        axes[1].set_title('$\phi$')
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/Theta2Phi.png')
        plt.close(fig)

        # Raw image and normalized image corrected for absorption
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        axes[0, 0].imshow(self.I_array, vmin=0, vmax=self.max_intensity)
        im0 = axes[0, 1].imshow(
            self.I_array / self.absorption_array,
            vmin=0, vmax=self.max_intensity
            )
        axes[1, 0].imshow(self.normalized_image_array, vmin=0.8, vmax=1.2)
        im1 = axes[1, 1].imshow(
            self.normalized_image_array / self.absorption_array,
            vmin=0.8, vmax=1.2
            )
        for row in range(2):
            for column in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        fig.colorbar(im0, ax=axes[0, 1])
        fig.colorbar(im1, ax=axes[1, 1])
        axes[0, 0].set_title('Initial')
        axes[0, 1].set_title('Corrected')
        axes[0, 0].set_ylabel('Raw Image')
        axes[1, 0].set_ylabel('Normalized Image')
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/Images.png')
        plt.close(fig)

        # 1D plots
        D = np.abs(self.s_array[:, :, 2])[np.invert(self.mask_array)].mean()
        delta = D * (self.h + self.t) / self.f
        alpha = self.angle
        b = delta / np.cos(alpha)
        m = -np.tan(alpha)

        min_pos = np.zeros(2)
        max_pos = np.zeros(2)
        for index in range(self.n_panels):
            origin = self.detector[index].get_origin()
            if origin[0] < min_pos[0]:
                min_pos[0] = origin[0]
            if origin[1] < min_pos[1]:
                min_pos[1] = origin[1]
            if origin[0] > max_pos[0]:
                max_pos[0] = origin[0]
            if origin[1] > max_pos[1]:
                max_pos[1] = origin[1]

        max_pos += np.array(self.panel_shape) * self.pixel_size
        y = np.array((0, max_pos[1] - min_pos[1]))
        y_pix = y / self.pixel_size
        b_pix = b / self.pixel_size
        x = m * (y + min_pos[1]) - b - min_pos[0]

        xx, yy = np.meshgrid(
            np.linspace(0, self.I_array.shape[1]*self.pixel_size, self.I_array.shape[1]),
            np.linspace(0, self.I_array.shape[0]*self.pixel_size, self.I_array.shape[0])
            )
        delta_x = (xx - (m * (yy + min_pos[1]) - b - min_pos[0])) * np.cos(alpha)

        delta_x_int = np.linspace(delta_x.min(), delta_x.max(), 100)
        bin_centers = (delta_x_int[1:] + delta_x_int[:-1]) / 2

        normalized_sum = np.histogram(
            delta_x[np.invert(self.mask_array)].ravel(),
            bins=delta_x_int,
            weights=self.normalized_image_array[np.invert(self.mask_array)].ravel()
            )
        absorption_sum = np.histogram(
            delta_x[np.invert(self.mask_array)].ravel(),
            bins=delta_x_int,
            weights=self.absorption_array[np.invert(self.mask_array)].ravel()
            )
        counts = np.histogram(
            delta_x[np.invert(self.mask_array)].ravel(),
            bins=delta_x_int
            )
        averaged_absorption = absorption_sum[0] / counts[0]
        averaged_normalized = normalized_sum[0] / counts[0]
        fig, axes = plt.subplots(1, 1)
        axes.plot(bin_centers, averaged_normalized, label='Normalized Image')
        axes.plot(bin_centers, averaged_absorption, label='Kapton Absorption')
        
        axes.plot(
            bin_centers, averaged_normalized / averaged_absorption,
            label='Corrected'
            )
        axes.set_xlabel('Perpendicual distance from maximum absorption (mm)')
        axes.set_ylabel('Intensity')
        axes.legend()
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/1DPlot.png')
        plt.close(fig)

        # Azimuthal Average
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.plot(
            180/np.pi * self.az_average[:, 0], self.az_average[:, 1],
            label='Flattened'
            )
        axes.plot(
            180/np.pi * self.az_average_array[:, 0], self.az_average_array[:, 1],
            label='Array'
            )
        axes.set_xlabel('$2\\theta$')
        axes.set_ylabel('Intensity')
        axes.legend()
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/AzAverage.png')
        plt.close(fig)
        return None

    def get_max_intensity(self):
        bins_raw_image = np.arange(0, self.max_intensity_limit, 0.1)
        centers_raw_image = (bins_raw_image[1:] + bins_raw_image[:-1]) / 2
        raw_image_width = bins_raw_image[1] - bins_raw_image[0]
        hist_raw_image, b = np.histogram(
            self.I_array[np.invert(self.mask_array)],
            bins=bins_raw_image,
            density=True
            )
        dI = bins_raw_image[1] - bins_raw_image[0]
        cummulative_sum = dI * hist_raw_image.cumsum()
        index = np.where(cummulative_sum > 0.9999)[0][0]
        self.max_intensity = centers_raw_image[index]
        return None

    def IDist(self, I_corrected, plot=False):
        bins = np.arange(self.max_intensity)
        
        theta2_bins = np.pi/180 * np.arange(25, 45, 1, dtype=np.float)
        theta2_centers = (theta2_bins[1:] + theta2_bins[:-1]) / 2
        n_theta2_bins = theta2_centers.size
        
        n_phi_bins = 16
        phi_bins = np.linspace(0, 2*np.pi, n_phi_bins + 1)
        phi_centers = (phi_bins[1:] + phi_bins[:-1]) / 2
        
        labels_theta2 = np.searchsorted(theta2_bins, self.theta2)
        labels_phi = np.searchsorted(phi_bins, self.phi)
        labels_all = 100 * labels_theta2 + labels_phi
       
        raw_pcc = np.zeros((n_theta2_bins, n_phi_bins))
        corrected_pcc = np.zeros((n_theta2_bins, n_phi_bins))  
        
        for theta2_index in range(n_theta2_bins):
            theta2_indices = labels_theta2 == theta2_index + 1
            I_corrected_theta2 = np.extract(theta2_indices, I_corrected)
            labels_all_theta2 = np.extract(theta2_indices, labels_all)
            
            corrected_all = numba_histogram(I_corrected_theta2, bins=bins)
            corrected_all = corrected_all / corrected_all.sum()
            for phi_index in range(n_phi_bins):
                label = 100 * (theta2_index + 1) + phi_index
                indices = labels_all_theta2 == label # Slow step
                if np.count_nonzero(indices) > 5000:
                    corrected = numba_histogram(
                        I_corrected_theta2[indices],
                        bins=bins
                        )
                    corrected = corrected / corrected.sum()
                    not_zero = corrected != 0    
                    corrected_pcc[theta2_index, phi_index] = np.corrcoef(
                        corrected_all[not_zero], corrected[not_zero]
                        )[0, 1]
                else:
                    corrected_pcc[theta2_index, phi_index] = np.nan

        mean_corrected_pcc = np.nanmean(corrected_pcc, axis=0)
        smallest_two = np.argsort(mean_corrected_pcc)[:2]
        mean_smallest_two = np.mean(mean_corrected_pcc[smallest_two])

        if plot:   
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            for theta2_index in range(n_theta2_bins):
                axes.plot(
                    180/np.pi * phi_centers,
                    corrected_pcc[theta2_index, :],
                    label='%2.2f' % (180/np.pi * theta2_centers[theta2_index])
                    )
            axes.plot(
                180/np.pi * phi_centers, mean_corrected_pcc,
                linewidth=4, color=[0, 0, 0]
                )
            ylim = axes.get_ylim()
            axes.plot([94, 94], ylim, color=[0, 0, 0], linestyle=':')
            axes.plot([274, 274], ylim, color=[0, 0, 0], linestyle=':')
            axes.set_xlabel('$\phi$')
            axes.set_ylabel('Correlation Coefficient')
            axes.set_ylim(ylim)
            axes.set_title(
                'Mean CC: %0.3f\nMean CC - smallest two: %0.3f'
                % (np.nanmean(mean_corrected_pcc), mean_smallest_two)
                )
            fig.tight_layout()
            fig.savefig(self.save_to_dir + '/Correlations.png')
            plt.close(fig)
        return mean_smallest_two

    def gradients(self, I_corrected, plot=False):
        def quick_interp(x, y, z, indices):
            shape = indices.shape
            interpx = np.zeros(shape)
            interpy = np.zeros(shape)
            for row in range(shape[0]):
                interpx[row, :] = np.interp(
                    y, y[indices[row, :]], z[row, indices[row, :]]
                    )
            for column in range(shape[1]):
                interpy[:, column] = np.interp(
                    x, x[indices[:, column]], z[indices[:, column], column]
                    )
            return (interpx + interpy) / 2
        
        bins = np.arange(self.max_intensity)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_widths = bins[1] - bins[0]
        
        theta2_bins = np.pi/180 * np.arange(5, 45, 1, dtype=np.float)
        theta2_centers = (theta2_bins[1:] + theta2_bins[:-1]) / 2
        n_theta2_bins = theta2_centers.size
        
        n_phi_bins = 64
        phi_bins = np.linspace(0, 2*np.pi, n_phi_bins + 1)
        phi_centers = (phi_bins[1:] + phi_bins[:-1]) / 2
        
        corrected_means, counts = numba_binned_mean(
            self.theta2, self.phi,
            bins=(theta2_bins, phi_bins),
            weights=I_corrected
            )

        indices = counts > 0
        corrected_means_interp = quick_interp(
            theta2_centers, phi_centers, corrected_means, indices
            )
        
        corrected_gradient_theta2, corrected_gradient_phi = np.abs(np.gradient(
            corrected_means_interp, theta2_centers, phi_centers
            ))
        R = np.tan(theta2_centers)
        for column in range(corrected_gradient_phi.shape[1]):
            corrected_gradient_phi[:, column] = R * corrected_gradient_phi[:, column]

        if plot:
            fig, axes = plt.subplots(
                1, 3, figsize=(10, 4),
                sharex=True, sharey=True
                )
            extent = [
                180/np.pi * phi_centers[0],
                180/np.pi * phi_centers[-1],
                180/np.pi * theta2_centers[-1],
                180/np.pi * theta2_centers[0]
                ]
            axes[0].imshow(corrected_means_interp, extent=extent, aspect='auto')
            axes[1].imshow(
                corrected_gradient_phi,
                vmin=0, vmax=0.9 * corrected_gradient_phi.max(),
                extent=extent, aspect='auto'
                )
            axes[2].imshow(
                corrected_gradient_theta2,
                vmin=0, vmax=0.9 * corrected_gradient_theta2.max(),
                extent=extent, aspect='auto'
                )
            axes[0].set_title('Corrected Image')
            axes[1].set_title(
                '$\phi$ gradient\nMean: %2.1f'
                % (corrected_gradient_phi[np.invert(indices)].mean())
                )
            axes[2].set_title(
                '$2\\theta$ gradient\nMean: %2.1f'
                % (corrected_gradient_theta2[np.invert(indices)].mean())
                )
            axes[0].set_ylabel('$2\\theta$')
            axes[0].set_xlabel('$\phi$')
            axes[1].set_xlabel('$\phi$')
            axes[2].set_xlabel('$\phi$')
            fig.tight_layout()
            fig.savefig(self.save_to_dir + '/Gradients.png')
            plt.close(fig)
        return corrected_gradient_phi[np.invert(indices)].mean()

    def _target_function_corr(self, params_in, phi_indices, s_norm_opt):
        params = [self.angle, *params_in]
        absorption = self.get_absorption_opt(s_norm_opt, phi_indices, *params)
        I_corrected = self.I / self.polarization / absorption
        mean_correlation = self.IDist(I_corrected)
        #print(params[0] * 180/np.pi)
        #print(params[1])
        #print(params[2])
        #print(mean_correlation)
        #print()
        return 1 / np.abs(mean_correlation)

    def _target_function_grad(self, params_in, phi_indices, s_norm_opt):
        params = [self.angle, *params_in]
        absorption = self.get_absorption_opt(s_norm_opt, phi_indices, *params)
        I_corrected = self.I / self.polarization / absorption
        mean_gradient = self.gradients(I_corrected)
        #print(params[0] * 180/np.pi)
        #print(params[1])
        #print(params[2])
        #print(mean_gradient)
        #print()
        return np.abs(mean_gradient)

    def _target_function_ltsq(self, params_in, phi_indices, s_norm_opt):
        params = [self.angle, *params_in]
        absorption = self.get_absorption_opt(s_norm_opt, phi_indices, *params)
        residuals = np.linalg.norm(self.normalized_image - absorption)
        #print(params[0])
        #print(params[1])
        #print(params[2])
        #print(residuals)
        #print()
        return np.linalg.norm(residuals)

    def _target_function_model_data(self, params_in, phi_indices, s_norm_opt):
        params = [self.angle, *params_in]
        absorption = self.get_absorption_opt(s_norm_opt, phi_indices, *params)
        difference = self.I * (1/self.normalized_image - 1/absorption)

        cost = np.linalg.norm(difference)
        #print(params[0])
        #print(params[1])
        #print(params[2])
        #print(cost)
        #print()
        return cost

    def _target_function_all(self, params_in, phi_indices, s_norm_opt):
        params = [self.angle, *params_in]
        absorption = self.get_absorption_opt(s_norm_opt, phi_indices, *params)
        I_corrected = self.I / self.polarization / absorption
        residuals = np.linalg.norm(self.normalized_image - absorption)
        correlation = self.IDist(I_corrected)
        gradient = self.gradients(I_corrected)
        target = 1/800 * residuals + 1/1.1 * 1/correlation + 1/35 * gradient
        #print(params[0])
        #print(params[1])
        #print(params[2])
        #print(residuals)
        #print(1/correlation)
        #print(gradient)
        #print(target)
        #print()
        return target

    def fit(self, target_function, method='L-BFGS-B'):
        if target_function == 'ltsq':
            tf = self._target_function_ltsq
        elif target_function == 'corr':
            tf = self._target_function_corr
        elif target_function == 'grad':
            tf = self._target_function_grad
        elif target_function == 'all':
            tf = self._target_function_all
        elif target_function == 'model - data':
            tf = self._target_function_model_data
        
        x0 = (self.h, self.f)
        bounds = (
            (0.001, None),
            (0.01, None),
            )
        phi_indices = np.logical_and(
            self.phi > np.pi/2 - self.angle,
            self.phi < 3*np.pi/2 - self.angle,
            )
        s_norm_opt = self.s_norm[phi_indices, :]
        self.fit_results = minimize(
            tf,
            x0=x0,
            method=method,
            bounds=bounds,
            args=(phi_indices, s_norm_opt)
            )
        #print(self.fit_results)
        return None

    def line_intensity(self, max_abs_x, max_abs_y, angle):
        # This calculates the mean intensity along a line defined by
        # max_abs_x, max_abs_y and angle. 
        y = np.arange(self.I_array.shape[0])
        x = np.tan(-angle)*(y-max_abs_y) + max_abs_x
        x = np.around(x).astype(np.int)
        indices = np.zeros(self.I_array.shape, dtype=np.bool)
        indices[y, x] = np.invert(self.mask_array[y, x])
        pixels = self.normalized_image_array[indices]
        intensities = self.I_array[indices]
        return pixels[intensities < self.max_intensity].mean()

    def initial_guess(self):
        def lorentzian_skew(x, x0, a, sl, su, m, b):
            curve = np.zeros(x.size)
            curve = m*x + b
            curve[x<x0] += a / (1 + ((x[x<x0]-x0)/sl)**2)
            curve[x>=x0] += a / (1 + ((x[x>=x0]-x0)/su)**2)
            return curve
        # This gets an initial guess for angle, h & f by finding
        # the point of maximum and minimum absorptin
        
        # Rough location near the point of maximum absorption
        #   max_abs_y - center of detector
        #   max_abs_x - point at maximum absorption
        #   angle - within 20 degrees
        max_abs_y = int(self.I_array.shape[0] / 2)
        max_abs_x = self.max_abs_point
        dx = 150
        new_angle = self.angle
        d_angle = 20 * np.pi/180
        
        max_abs_x_array = np.arange(max_abs_x - dx, max_abs_x + dx, 1)
        
        divisors = [1, 2, 3, 5]

        for d in divisors:
            angle_array = np.arange(
                new_angle - d_angle / d,
                new_angle + d_angle / d + 0.1*np.pi/180,
                0.1*np.pi/180)
            meanI_angle = np.zeros(angle_array.size)
            for index, angle in enumerate(angle_array):
                meanI_angle[index] = self.line_intensity(
                    max_abs_x, max_abs_y, angle
                    )

            # To find the correct angle - a curve is fit to this mean intensity
            # A two-sided lorentzian function is fit to the curve
            popt, pcov = curve_fit(
                lorentzian_skew, angle_array, meanI_angle,
                p0=(new_angle, -1, 0.01, 0.01,  0, 1)
                )
            new_angle = popt[0]
            # Do a similar procedure for the max_abs_x location
            if d in divisors[:2]:
                meanI_x = np.zeros(max_abs_x_array.size)
                for index, x in enumerate(max_abs_x_array):
                    meanI_x[index] = self.line_intensity(
                        x, max_abs_y, new_angle
                        )
                y = gaussian_filter1d(medfilt(meanI_x, 21), 5)
                max_abs_x = max_abs_x_array[np.argmin(y)]
        
        y = gaussian_filter1d(medfilt(meanI_x, 1), 11)
        der = np.gradient(y, max_abs_x_array)
        der2 = np.gradient(der, max_abs_x_array)
        min_absorption_index = np.argmin(der2)
        max_absorption_index = np.argmax(der2)

        #dx_max = self.beam_center_x - max_abs_x
        #dx_min = self.beam_center_x - max_abs_x_array[min_absorption_index]

        #A = self.pixel_size / np.abs(self.s[:, 2].mean()) * np.cos(np.pi/180 * new_angle)
        #h = self.t / (dx_max/dx_min - 1)
        #f = h / (A * dx_min)

        y = np.array([0, self.I_array.shape[0]])
        x = np.tan(-new_angle)*(y-max_abs_y) + max_abs_x
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.imshow(
            self.normalized_image_array[:, 1800: 2200],
            vmin=0.6, vmax=1
            )
        axes.plot(x - 1800, y)
        fig.tight_layout()
        fig.savefig(self.save_to_dir + '/LineDrawing.png')
        plt.close(fig)
        return new_angle
