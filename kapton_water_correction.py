"""
Test kapton model with tape at 45 degrees
Setup code so the model works for just modeling single frames
Speed up setup with multiprocessing

fix runtime warnings

# time estimates #
total time
   30 s

- loading data
   0.1 s

- setting up image
   3.5 s

   - setting up image
      1 s

   - setting up binned image
      1.5 s

   - normalizing images
      1 s

- fitting absorption model
   7.5 s

- full absorption model calculation
   5 s

- uncertainty calculation
   10 s

image setup
    - make large image without padding between panels. Create a function that 
        returns a spaced image for plotting purposes
    - 1d resentation with ravel

Uncertainty calculation
    move rankit calculation out of for loop

absorption optimization
    model initializations
    first derivative calculation

absorption model calculation
    - GPU
    - convert to c++

"""
from __future__ import annotations

from cctbx import factor_kev_angstrom
from dials.array_family import flex
import dxtbx
from dxtbx import flumpy
from iotbx import phil
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numba
import os
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.optimize.optimize import MemoizeJac
from scipy.optimize import minimize
from scipy.signal import medfilt
from scipy.signal import medfilt2d
import scipy.stats


#matplotlib.use('Agg')


absorption_defs = """
  absorption_correction
    .multiple = True {
    apply = False
      .type = bool
      .help = must be supplied as a user-defined function with a specific \
              interface (not documented)
    algorithm = fuller_kapton kapton_2019 kapton_water other
      .type = choice
      .help = a specific absorption correction, or implementation thereof \
              kapton_2019 is a more general implementation of fuller_kapton \
              for use on single/multi-panel detectors. kapton_water also \
              corrects for water absorption
    kapton_water {
      model {
        value = 'both'
          .type = str
          .help = Possible choices: 'kapton', 'water', 'both'.\
                  Determines if kapton, water, or both absorption corrections \
                  are applied.
        }
      method {
        value = 'fit'
          .type = bool
          .help = 'fit': fit absorption model to each image.\
                  'fit average': fit absorption model to the average image \
                    and apply this to each frame. There will be an average \
                    image for each detector contained in the expt file.
                  'constant': use constant model based on input parameters,\
        }
      height_above_kapton_mm {
        value = 0.02
          .type = float
          .help = Distance of the beam in the direction normal to the kapton \
                  tape. Units are in mm.
        }
      rotation_angle_deg {
        value = 0
          .type = float
          .help = Angle of the tape from vertical. Units are in degrees
        }
      forward_distance_mm {
        value = 1
          .type = float
          .help = Distance between the center of the droplet to edge of tape \
                  nearest detector. Units are in mm.
        }
      kapton_thickness_mm {
        value = 0.025
          .type = float
          .help = Kapton tape thickness. Units are in mm.
        }
      droplet_volume_ul {
        value = 0.2
          .type = float
          .help = Volume of droplet dispensed onto the kapton film. \
                  Units are in microliters
        }
      droplet_contact_angle_deg {
        value = 45
          .type = float
          .help = Contact angle of droplet dispensed onto the kapton film. \
                  Units are in degrees
        }
      parallel_distance_fraction {
        value = 0
          .type = float
          .help = Offset of the beam from the center of the droplet along the \
                  length of the kapton tape. Units are fraction of the \
                  maximum possible distance.
        }
      scale {
        value = 0.95
          .type = float
          .help = scaling used to match the absorption model to the \
                  normalized image during optimization.
        }
      pedestal_rms_name {
        value = None
          .type = str
          .help = file name for the pedestal_rms file. This is used to \
                  calculate the uncertainty of the absorption correction.
        }
      binning {
        value = 16
          .type = int
          .help = image size is reduced for optimization by averaging pixels \
                  in binning x binning squares.
        }
      pad {
        value = 2
          .type = int
          .help = number of pixels to mask at the panel edges.
        }
      bad_panels {
        value = None
          .type = str
          .help = list of panels that should be masked out
        }
      theta2_start {
        value = 0
          .type = float
          .help = start of theta2 angles used for optimization and uncertainty.
        }
      theta2_stop {
        value = 60
          .type = float
          .help = stop of theta2 angles used for optimization and uncertainty.
        }
      theta2_step {
        value = 1
          .type = float
          .help = step of theta2 angles used for optimization and uncertainty.
        }
      theta2_stride_err {
        value = 4
          .type = int
          .help = parameter specifying the spacing in 2theta for the \
                  uncertainty calculation.
        }
      n_phi_bins_err {
        value = 128
          .type = int
          .help = parameter specifying the number of bins in phi for the \
                  uncertainty calculation.
        }
    }
  }"""


#absorption_phil_scope = phil.parse(absorption_defs, process_includes=True)


@numba.jit(nopython=True)
def numba_binned_mean(a, bins, weights):
    n = bins.size - 1
    a_min = bins[0]
    a_max = bins[-1]
    factor = n / (a_max - a_min)
    summed = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.intp)
    averaged = np.zeros(n, dtype=np.float64)
    for index, x in enumerate(a):
        bin = int(factor * (x - a_min))
        if bin >= 0 and bin < n:
            counts[bin] += 1
            summed[bin] += weights[index]
    averaged = summed / counts
    return averaged, counts


@numba.jit(nopython=True)
def numba_histogram(a, bins):
    n = bins.size - 1
    a_min = bins[0]
    a_max = bins[-1]
    factor = n / (a_max - a_min)
    hist = np.zeros(n, dtype=np.intp)
    for x in a:
        bin = int(factor * (x - a_min))
        if bin >= 0 and bin < n:
            hist[bin] += 1
    return hist


class detector_setup:
    def __init__(self, imageset, params):
        detector = imageset.get_detector()

        self.wavelength = imageset.get_beam().get_wavelength()
        self.energy = factor_kev_angstrom / self.wavelength
        self._get_absorption_lengths()

        if params['use_pedestal_rms']:
            if params['pedestal_rms_name'] == 'h5 image':
                pedestal_rms = self._get_pedestalRMS_from_jungfrau(imageset)
            else:
                pedestal_rms = \
                    np.load(params['pedestal_rms_name']) / self.energy
        else:
            pedestal_rms = None

        self.plots = params['plots']
        self.bad_panels = params['bad_panels']
        self.binning = params['binning']
        self.pad = params['pad']
        self.polarization_fraction = params['polarization_fraction']

        self.theta2_start = np.pi/180 * params['theta2_start_deg']
        self.theta2_stop = np.pi/180 * params['theta2_stop_deg']
        self.theta2_step = np.pi/180 * params['theta2_step_deg']
        self.theta2_bins = np.arange(
            self.theta2_start,
            self.theta2_stop + self.theta2_step,
            self.theta2_step
            )
        self.n_theta2_bins = self.theta2_bins.size
        self.theta2_start_err = 0.95 * params['theta2_err_lims'][0]
        self.theta2_stop_err = 1.05 * params['theta2_err_lims'][1]
        self.theta2_step_err = np.pi/180 * params['theta2_step_err_deg']
        self.n_phi_bins_err = params['n_phi_bins_err']
        self.n_pixels_phi_bins_err = params['n_pixels_phi_bins_err']

        self.n_panels = len(detector)
        self.pixel_size = detector[0].get_pixel_size()[0]
        self.panel_shape = detector[0].get_image_size()
        self.panel_size = self.panel_shape[0] * self.panel_shape[1]
        self.n_pixels_full = self.n_panels * self.panel_size

        ###!!! Binning needs to be fixed so it doesn't assume a 254 pixel
        ###!!! square that I manually pad to make it 256...
        self.binned_pixel_size = self.pixel_size * self.binning
        self.binned_panel_shape = [
            int((self.panel_shape[0] + 2) / self.binning),
            int((self.panel_shape[1] + 2) / self.binning)
            ]
        self.binned_panel_size = \
            self.binned_panel_shape[0] * self.binned_panel_shape[1]
        self.binned_n_pixels_full = self.n_panels * self.binned_panel_size

        self._get_s1_mask(detector, pedestal_rms)
        self._get_s1_binned(detector)
        self._setup_uncertainty()
        return None

    def _setup_process_panel(self, panel, x, y):
        # This returns information about an individual panel to be fed into
        # a single array to represent the entire image
        # s1 - vector from crystal to each pixel
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(x.ravel()),
                    flex.double(y.ravel())
                )))
            )

        mask = np.zeros(self.panel_shape, dtype=bool)
        # Mask out the edges of the panel
        mask[:self.pad, :] = True
        mask[-self.pad:, :] = True
        mask[:, :self.pad] = True
        mask[:, -self.pad:] = True
        mask = mask.ravel()
        return s1, mask

    def _get_s1_mask(self, detector, pedestal_rms):
        x, y = np.meshgrid(
            np.arange(self.panel_shape[0]),
            np.arange(self.panel_shape[1])
            )

        self._min_pos = np.zeros(2)
        max_pos = np.zeros(2)
        for index, panel in enumerate(detector):
            origin = panel.get_origin()
            if origin[0] < self._min_pos[0]:
                self._min_pos[0] = origin[0]
            if origin[1] < self._min_pos[1]:
                self._min_pos[1] = origin[1]
            if origin[0] > max_pos[0]:
                max_pos[0] = origin[0]
            if origin[1] > max_pos[1]:
                max_pos[1] = origin[1]
        max_pos = np.rint(
            (max_pos - self._min_pos) / self.pixel_size
            ).astype(int)
        max_pos += self.panel_shape
        self.image_shape = max_pos[::-1]

        if pedestal_rms is None:
            self.pedestal_rms = None
        else:
            self.pedestal_rms = -1*np.ones(self.image_shape)

        self.s1 = -1*np.ones((*self.image_shape, 3)).ravel()
        self.mask = np.ones(self.image_shape, dtype=bool)
        self.panel_origin = np.zeros((self.n_panels, 2), dtype=int)
        for panel_index, panel in enumerate(detector):
            self.panel_origin[panel_index, :] = np.rint(
                (panel.get_origin()[:2] - self._min_pos) / self.pixel_size
                ).astype(int)
            s1_panel, mask_panel = self._setup_process_panel(panel, x, y)
            indices = np.zeros(max_pos[::-1], dtype=bool)
            indices[
                self.panel_origin[panel_index, 1]:
                self.panel_origin[panel_index, 1] + self.panel_shape[1],
                self.panel_origin[panel_index, 0]:
                self.panel_origin[panel_index, 0] + self.panel_shape[0]
                ] = True
            indices_flat = np.zeros((*max_pos[::-1], 3), dtype=bool)
            indices_flat[
                self.panel_origin[panel_index, 1]:
                self.panel_origin[panel_index, 1] + self.panel_shape[1],
                self.panel_origin[panel_index, 0]:
                self.panel_origin[panel_index, 0] + self.panel_shape[0],
                :
                ] = True

            indices_flat = indices_flat.ravel()

            if self.pedestal_rms is not None:
                self.pedestal_rms[indices] = pedestal_rms[index, :, :].ravel()

            self.mask[indices] = mask_panel
            self.s1[indices_flat] = s1_panel.ravel()
            if panel_index in self.bad_panels:
                self.mask[indices] = True

        self.s1 = self.s1.reshape(
            (self.image_shape[0] * self.image_shape[1], 3)
            )
        self.s1_norm = (self.s1.T / np.linalg.norm(self.s1, axis=1)).T
        self.s1 = self.s1.reshape((*self.image_shape, 3))
        self.s1_norm = self.s1_norm.reshape((*self.image_shape, 3))

        # s1 is defined with the positive z axis pointing from the detector to
        # the crystal. The code relies on the vector pointing from the crystal
        # to the detector 
        self.s1_norm[:, :, 2] *= -1

        self.theta2, self.phi = self.get_theta2_phi(self.s1)
        self.theta2[self.mask] = -1
        self.phi[self.mask] = -1
        self.polarization = self._get_polarization(self.theta2, self.phi)
        self.polarization[self.mask] = -1
        return None

    def _setup_process_panel_binned(self, panel, x, y):
        s1 = flumpy.to_numpy(
                panel.get_lab_coord(panel.pixel_to_millimeter(flex.vec2_double(
                    flex.double(x.ravel()),
                    flex.double(y.ravel())
                )))
            )

        s1_norm = (s1.T / np.linalg.norm(s1, axis=1)).T
        s1_norm[:, 2] *= -1
        return s1, s1_norm

    def _get_s1_binned(self, detector):
        x, y = np.meshgrid(
            np.arange(
                int(self.binning / 2),
                self.panel_shape[0] + 2 + int(self.binning / 2),
                self.binning
                ),
            np.arange(
                int(self.binning / 2),
                self.panel_shape[1] + 2 + int(self.binning / 2),
                self.binning
                )
            )
        self.s1_bin = np.zeros((self.binned_n_pixels_full, 3))
        self.s1_norm_bin = np.zeros((self.binned_n_pixels_full, 3))
        for index, panel in enumerate(detector):
            if index not in self.bad_panels:
                s1_panel, s1_norm_panel = \
                    self._setup_process_panel_binned(panel, x, y)
                start = index * self.binned_panel_size
                end = (index + 1) * self.binned_panel_size
                self.s1_bin[start: end] = s1_panel
                self.s1_norm_bin[start: end] = s1_norm_panel
        self.theta2_bin, self.phi_bin = self.get_theta2_phi(self.s1_bin)
        self.polarization_bin = self._get_polarization(
            self.theta2_bin, self.phi_bin
            )
        return None

    def _get_polarization(self, theta2, phi):
        fx = self.polarization_fraction
        cos_theta2_2 = np.cos(theta2)**2
        cos_phi_2 = np.cos(phi)**2
        sin_phi_2 = np.sin(phi)**2
        fy = 1 - fx
        Px = fx * (sin_phi_2 + cos_phi_2 * cos_theta2_2)
        Py = fy * (cos_phi_2 + sin_phi_2 * cos_theta2_2)
        return Px + Py

    def get_theta2_phi(self, s):
        ###!!! I am plotting the images upside down Phi should be updated
        # If working with 2D arrays
        if len(s.shape) == 3:
            s0 = s[:, :, 0]
            s1 = s[:, :, 1]
            s2 = s[:, :, 2]
        # If working with flattened arrays
        else:
            s0 = s[:, 0]
            s1 = s[:, 1]
            s2 = s[:, 2]
        R = np.sqrt(s0**2 + s1**2)
        theta2 = np.zeros(s2.shape)
        indices = s2 != 0
        theta2[indices] = np.arctan(R[indices] / np.abs(s2[indices]))
        theta2[np.invert(indices)] = -1
        phi = np.pi + np.arctan2(s1, -1*s0)
        return theta2, phi

    def _setup_uncertainty(self):
        self.n_pixels_phi_bins_err
        self.theta2_bins_err = np.arange(
            self.theta2_start_err,
            self.theta2_stop_err + self.theta2_step_err,
            self.theta2_step_err
            )

        self.theta2_centers_err = \
            (self.theta2_bins_err[1:] + self.theta2_bins_err[:-1]) / 2
        self.n_theta2_bins_err = self.theta2_centers_err.size

        n_pixels = numba_histogram(self.theta2[np.invert(self.mask)], self.theta2_bins_err)
        self.n_phi_bins_err = np.rint(n_pixels / self.n_pixels_phi_bins_err).astype(int)
        self.n_bins = self.n_phi_bins_err.sum()

        # search sorted
        # 0: below range
        # 1 -> n: in bins
        # n + 1: above range
        labels_theta2 = \
            np.searchsorted(self.theta2_bins_err, self.theta2.ravel())

        self.phi_bins_err = []
        self.phi_centers = []
        self.start_stop = []
        labels_phi = np.zeros(labels_theta2.shape)
        for index in range(self.n_theta2_bins_err):
            phi_bins_err = np.linspace(0, 2*np.pi, self.n_phi_bins_err[index] + 1)
            phi_centers = (phi_bins_err[1:] + phi_bins_err[:-1]) / 2
            self.phi_bins_err.append(phi_bins_err)
            self.phi_centers.append(phi_centers)
            indices = labels_theta2 == index + 1
            labels_phi_here = np.searchsorted(phi_bins_err, self.phi.ravel())
            labels_phi[indices] += labels_phi_here[indices]
            self.start_stop.append(
                np.zeros(self.n_phi_bins_err[index] + 1, dtype=int)
                )

        labels = self.n_phi_bins_err.max() * (labels_theta2 - 1) + (labels_phi - 1)
        self.sort_indices = np.argsort(labels).astype(int)
        labels.sort()

        for theta2_index in range(self.n_theta2_bins_err):
            label_min = self.n_phi_bins_err.max() * theta2_index
            label_max = self.n_phi_bins_err.max() * (theta2_index + 1)
            indices = np.nonzero(np.logical_and(
                labels >= label_min,
                labels < label_max
                ))[0]
            if indices.size > 0:
                labels_theta2 = labels[indices]
                start_theta2 = indices.min()
                stop_theta2 = indices.max() + 1
                self.start_stop[theta2_index][0] = start_theta2
                self.start_stop[theta2_index][-1] = stop_theta2
            for phi_index in range(self.n_phi_bins_err[theta2_index]):
                label = self.n_phi_bins_err.max() * theta2_index + phi_index
                indices_phi = np.nonzero(labels_theta2 == label)[0]
                if indices_phi.size > 0:
                    self.start_stop[theta2_index][phi_index] = \
                        indices_phi.min() + start_theta2
                    self.start_stop[theta2_index][phi_index + 1] = \
                        indices_phi.max() + 1 + start_theta2
        return None

    def _get_pedestalRMS_from_jungfrau(self, imageset):
        F = imageset.get_format_class()
        fclass = F.get_instance(imageset.paths()[0])
        pedestal_rms = fclass.get_pedestal_rms(
            imageset.indices()[0],
            return_gain_modes=False
            )
        return pedestal_rms / self.energy

    def _get_absorption_lengths(self):
        # energy (MeV), mu/rho (cm^2/g)
        # https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
        water_data = np.array([
            [1.00000E-03,  4.078E+03],
            [1.50000E-03,  1.376E+03],
            [2.00000E-03,  6.173E+02],
            [3.00000E-03,  1.929E+02],
            [4.00000E-03,  8.278E+01],
            [5.00000E-03,  4.258E+01],
            [6.00000E-03,  2.464E+01],
            [8.00000E-03,  1.037E+01],
            [1.00000E-02,  5.329E+00],
            [1.50000E-02,  1.673E+00],
            [2.00000E-02,  8.096E-01],
            [3.00000E-02,  3.756E-01],
            [4.00000E-02,  2.683E-01],
            [5.00000E-02,  2.269E-01],
            [6.00000E-02,  2.059E-01],
            [8.00000E-02,  1.837E-01],
            [1.00000E-01,  1.707E-01],
            [1.50000E-01,  1.505E-01],
            [2.00000E-01,  1.370E-01],
            [3.00000E-01,  1.186E-01],
            [4.00000E-01,  1.061E-01],
            [5.00000E-01,  9.687E-02],
            [6.00000E-01,  8.956E-02],
            [8.00000E-01,  7.865E-02],
            [1.00000E+00,  7.072E-02],
            [1.25000E+00,  6.323E-02],
            [1.50000E+00,  5.754E-02],
            [2.00000E+00,  4.942E-02],
            [3.00000E+00,  3.969E-02],
            [4.00000E+00,  3.403E-02],
            [5.00000E+00,  3.031E-02],
            [6.00000E+00,  2.770E-02],
            [8.00000E+00,  2.429E-02],
            [1.00000E+01,  2.219E-02],
            [1.50000E+01,  1.941E-02],
            [2.00000E+01,  1.813E-02]
            ])
        water_density = 0.99777  # g/cm^3
        water_data[:, 0] *= 10**3
        water_absorption_coefficient = \
            np.interp(self.energy, water_data[:, 0], water_data[:, 1])
        self.water_absorption_length = \
            10 / (water_density * water_absorption_coefficient)

        # energy (ev), absorption_length (um)
        kapton_data = np.array([
            [6000.00,  482.643],
            [6070.00,  500.286],
            [6140.00,  518.362],
            [6210.00,  536.896],
            [6280.00,  555.873],
            [6350.00,  575.302],
            [6420.00,  595.191],
            [6490.00,  615.552],
            [6560.00,  636.382],
            [6630.00,  657.691],
            [6700.00,  679.484],
            [6770.00,  701.758],
            [6840.00,  724.521],
            [6910.00,  747.791],
            [6980.00,  771.561],
            [7050.00,  795.846],
            [7120.00,  820.646],
            [7190.00,  845.963],
            [7260.00,  871.812],
            [7330.00,  898.183],
            [7400.00,  925.082],
            [7470.00,  952.535],
            [7540.00,  980.535],
            [7610.00,  1009.09],
            [7680.00,  1038.18],
            [7750.00,  1067.85],
            [7820.00,  1098.08],
            [7890.00,  1128.88],
            [7960.00,  1160.25],
            [8030.00,  1192.20],
            [8100.00,  1224.76],
            [8170.00,  1257.91],
            [8240.00,  1291.67],
            [8310.00,  1326.01],
            [8380.00,  1360.98],
            [8450.00,  1396.54],
            [8520.00,  1432.72],
            [8590.00,  1469.51],
            [8660.00,  1506.93],
            [8730.00,  1544.96],
            [8800.00,  1583.65],
            [8870.00,  1622.95],
            [8940.00,  1662.90],
            [9010.00,  1703.49],
            [9080.00,  1744.72],
            [9150.00,  1786.59],
            [9220.00,  1829.13],
            [9290.00,  1872.31],
            [9360.00,  1916.16],
            [9430.00,  1960.65],
            [9500.00,  2005.82],
            [9570.00,  2051.65],
            [9640.00,  2098.16],
            [9710.00,  2145.36],
            [9780.00,  2193.22],
            [9850.00,  2241.75],
            [9920.00,  2290.95],
            [9990.00,  2340.86],
            [10060.0,  2391.49],
            [10130.0,  2442.84],
            [10200.0,  2494.86],
            [10270.0,  2547.59],
            [10340.0,  2601.02],
            [10410.0,  2655.14],
            [10480.0,  2709.98],
            [10550.0,  2765.49],
            [10620.0,  2821.73],
            [10690.0,  2878.68],
            [10760.0,  2936.31],
            [10830.0,  2994.67],
            [10900.0,  3053.72],
            [10970.0,  3113.49],
            [11040.0,  3173.96],
            [11110.0,  3235.14],
            [11180.0,  3297.07],
            [11250.0,  3359.67],
            [11320.0,  3423.01],
            [11390.0,  3487.04],
            [11460.0,  3551.76],
            [11530.0,  3617.23],
            [11600.0,  3683.38],
            [11670.0,  3750.23],
            [11740.0,  3817.81],
            [11810.0,  3886.07],
            [11880.0,  3955.05],
            [11950.0,  4024.75],
            [12020.0,  4095.11],
            [12090.0,  4166.20],
            [12160.0,  4237.96],
            [12230.0,  4310.40],
            [12300.0,  4383.60],
            [12370.0,  4457.48],
            [12440.0,  4532.02],
            [12510.0,  4607.25],
            [12580.0,  4683.14],
            [12650.0,  4759.73],
            [12720.0,  4837.01],
            [12790.0,  4914.94],
            [12860.0,  4993.54],
            [12930.0,  5072.79],
            [13000.0,  5152.69],
            ])
        kapton_data[:, 0] *= 10**-3
        kapton_data[:, 1] *= 10**-3
        self.kapton_absorption_length = \
            np.interp(self.energy, kapton_data[:, 0], kapton_data[:, 1])
        return None


class MemoizeJacHess(MemoizeJac):
    """ Decorator that caches the return vales of a function returning
        (fun, grad, hess) each time it is called. """

    def __init__(self, fun):
        super().__init__(fun)
        self.hess = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None or self.hess is None:
            self.x = np.asarray(x).copy()
            self._value, self.jac, self.hess = self.fun(x, *args)

    def hessian(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.hess


class absorption_model:
    def __init__(self, setup):
        self.water_absorption_length = setup.water_absorption_length
        self.kapton_absorption_length = setup.kapton_absorption_length
        self.plots = setup.plots
        return None

    def load_results(self, frame, params):
        self.frame = frame
        self.save_to_dir = \
            params['save_to_dir'] + '/Frame_' + str(self.frame).zfill(5)
        if os.path.exists(self.save_to_dir):
            absorption_params = np.loadtxt(
                self.save_to_dir
                + '/Params_' + str(self.frame).zfill(5) + '.txt'
                )
            self.angle = absorption_params[0]
            self.h = absorption_params[1]
            self.f = absorption_params[2]
            self.vf = absorption_params[3]
            self.volume = absorption_params[4]
            self.contact_angle = absorption_params[5]
            self.scale = absorption_params[6]
            self.t = absorption_params[7]
            
            self.err_array = np.load(
                self.save_to_dir
                + '/err_array_' + str(self.frame).zfill(5) + '.npy',
                )
            self.theta2_bins_err = np.load(
                self.save_to_dir
                + '/theta2_bins_err_' + str(self.frame).zfill(5) + '.npy',
                )
            self.phi_bins_err = np.load(
                self.save_to_dir
                + '/phi_bins_err_' + str(self.frame).zfill(5) + '.npy',
                )
            self.n_phi_bins_err = self.phi_bins_err.size
            self.n_theta2_bins_err = self.theta2_bins_err.size
            found = True
        else:
            found = False
        return found

    def update_frame(self, data, frame, params, refls, setup):
        self.frame = frame
        self.save_to_dir = \
            params['save_to_dir'] + '/Frame_' + str(self.frame).zfill(5)
        if os.path.exists(self.save_to_dir) is False:
            os.mkdir(self.save_to_dir)
        self.angle = np.pi/180 * params['rotation_angle_deg']
        self.h = params['height_above_kapton_mm']
        self.f = params['forward_distance_mm']
        self.t = params['kapton_thickness_mm']
        self.vf = params['parallel_distance_fraction']
        self.volume = params['droplet_volume_ul']
        self.contact_angle = np.pi/180 * params['droplet_contact_angle_deg']
        self.scale = params['scale']
        self.max_abs_point = params['max_abs_point']
        self.mask = np.copy(setup.mask)
        self.get_image(data, setup, refls)
        self.get_image_binned(setup)
        self.get_normalized_images(setup)
        self.phi_bins_err = setup.phi_bins_err
        self.theta2_bins_err = setup.theta2_bins_err
        self.n_phi_bins_err = setup.n_phi_bins_err
        self.n_theta2_bins_err = setup.n_theta2_bins_err
        return None

    def get_image(self, data, setup, refls):
        self.I = -1*np.ones(setup.image_shape)
        for panel_index in range(setup.n_panels):
            I_panel = flumpy.to_numpy(data[panel_index]) / setup.energy
            self.I[
                setup.panel_origin[panel_index, 1]:
                setup.panel_origin[panel_index, 1] + setup.panel_shape[1],
                setup.panel_origin[panel_index, 0]:
                setup.panel_origin[panel_index, 0] + setup.panel_shape[0]
                ] = I_panel

            if panel_index in setup.bad_panels:
                mask_panel = np.ones(setup.panel_shape, dtype=bool)
            else:
                mask_panel = np.zeros(setup.panel_shape, dtype=bool)
                mask_panel[:setup.pad, :] = True
                mask_panel[-setup.pad:, :] = True
                mask_panel[:, :setup.pad] = True
                mask_panel[:, -setup.pad:] = True

                refls_panel = refls.select(refls['panel'] == panel_index)
                if len(refls_panel) > 0:
                    for refls_index in range(len(refls_panel)):
                        mask_panel[
                            refls_panel[refls_index]['bbox'][2]:
                            refls_panel[refls_index]['bbox'][3],
                            refls_panel[refls_index]['bbox'][0]:
                            refls_panel[refls_index]['bbox'][1],
                            ] = True

                #I_panel_filtered = medfilt2d(
                #    I_panel,
                #    kernel_size=(5, 5)
                #    )
                # z = np.abs((I_panel - I_panel_filtered) / I_panel.std())
                z = np.abs((I_panel - np.median(I_panel)) / I_panel.std())
                mask_panel[z > 5] = True

                self.mask[
                    setup.panel_origin[panel_index, 1]:
                    setup.panel_origin[panel_index, 1] + setup.panel_shape[1],
                    setup.panel_origin[panel_index, 0]:
                    setup.panel_origin[panel_index, 0] + setup.panel_shape[0]
                    ] = mask_panel

        self.mask[self.I <= 0] = True
        self.mask_inv = np.invert(self.mask)
        return None

    def _process_panel_binned(self, setup, panel_index):
        I = self.I[
            setup.panel_origin[panel_index, 1]:
            setup.panel_origin[panel_index, 1] + setup.panel_shape[1],
            setup.panel_origin[panel_index, 0]:
            setup.panel_origin[panel_index, 0] + setup.panel_shape[0]
            ]
        mask = self.mask[
            setup.panel_origin[panel_index, 1]:
            setup.panel_origin[panel_index, 1] + setup.panel_shape[1],
            setup.panel_origin[panel_index, 0]:
            setup.panel_origin[panel_index, 0] + setup.panel_shape[0]
            ]

        I = np.column_stack((np.zeros(I.shape[0]), I, np.zeros(I.shape[0])))
        I = np.row_stack((np.zeros(I.shape[1]), I, np.zeros(I.shape[1])))

        mask = np.column_stack((
            np.ones(mask.shape[0], dtype=bool),
            mask,
            np.ones(mask.shape[0], dtype=bool)
            ))
        mask = np.row_stack((
            np.ones(mask.shape[1], dtype=bool),
            mask,
            np.ones(mask.shape[1], dtype=bool)
            ))

        I[mask] = np.nan
        I_reshaped = I.reshape((
            int(I.shape[0] / setup.binning),
            setup.binning,
            int(I.shape[1] / setup.binning),
            setup.binning
            ))

        # median does a lot to remove unmasked bragg peaks
        I_binned = np.nanmedian(np.nanmedian(I_reshaped, axis=3), axis=1)

        mask_binned_sum = mask.reshape((
            int(I.shape[0] / setup.binning),
            setup.binning,
            int(I.shape[1] / setup.binning),
            setup.binning
            )).sum(axis=3).sum(axis=1)

        mask_binned = mask_binned_sum == setup.binning**2
        return I_binned.ravel(), mask_binned.ravel()

    def get_image_binned(self, setup):
        self.I_bin = np.zeros(setup.binned_n_pixels_full)
        mask_bin = np.zeros(setup.binned_n_pixels_full, dtype=bool)
        for panel_index in range(setup.n_panels):
            if panel_index in setup.bad_panels:
                mask_bin[start: end] = True
            else:
                I_panel, mask_panel = \
                    self._process_panel_binned(setup, panel_index)
                start = panel_index * setup.binned_panel_size
                end = (panel_index + 1) * setup.binned_panel_size
                self.I_bin[start: end] = I_panel
                mask_bin[start: end] = mask_panel
        mask_bin[self.I_bin <= 0] = True
        self.binned_n_pixels_full = np.invert(mask_bin).sum()
        self.I_bin = np.delete(self.I_bin, mask_bin)
        self.s1_bin = np.delete(setup.s1_bin, mask_bin, axis=0)
        self.s1_norm_bin = np.delete(setup.s1_norm_bin, mask_bin, axis=0)
        self.theta2_bin = np.delete(setup.theta2_bin, mask_bin)
        self.phi_bin = np.delete(setup.phi_bin, mask_bin)
        self.polarization_bin = np.delete(setup.polarization_bin, mask_bin)
        return None

    def get_normalized_images(self, setup):
        # Normalizes the binned image
        self.az_average_bin = self._azimuthal_average(
            self.theta2_bin,
            (self.I_bin / self.polarization_bin)
            )
        self.integrated_image_bin = np.interp(
            self.theta2_bin,
            self.az_average_bin[:, 0],
            self.az_average_bin[:, 1]
            )
        self.normalized_image_bin \
            = self.I_bin / (self.polarization_bin * self.integrated_image_bin)

        # Normalizes the arrayed image
        I_pol = self.I / setup.polarization
        self.az_average = self._azimuthal_average(
            setup.theta2[self.mask_inv].ravel(),
            I_pol[self.mask_inv].ravel()
            )
        self.integrated_image = np.interp(
            setup.theta2,
            self.az_average[:, 0],
            self.az_average[:, 1]
            )
        self.normalized_image = I_pol / self.integrated_image
        self.integrated_image[self.mask] = -1
        self.normalized_image[self.mask] = -1
        return None

    def _azimuthal_average(self, theta2, I):
        # Returns the azimuthally averaged image
        # The average is calculated with histograms - very fast implementation
        #   counts == number of pixels in a bin
        #   sum == summed intensity in a bin
        #   sum / counts == average pixel intensity
        theta2_bins = np.linspace(0, 60, 61) * np.pi/180
        theta2_centers = (theta2_bins[1:] + theta2_bins[:-1]) / 2
        integrated, integration_counts = numba_binned_mean(
            theta2, bins=theta2_bins, weights=I
            )
        indices = integration_counts > 0
        az_average = np.column_stack(
            (theta2_centers[indices], integrated[indices])
            )
        return az_average

    def fit_absorption_model(self, setup):
        print('Angle')
        self.angle = self.initial_guess()
        print(180/np.pi * self.angle)
        print()
        
        print('water')
        fit_results = self.fit(model='water')
        self.h = fit_results.x[0]
        self.vf = fit_results.x[1]
        self.volume = fit_results.x[2]
        self.contact_angle = fit_results.x[3]
        self.scale = fit_results.x[4]
        print(fit_results)
        print(self.h)
        print(self.vf)
        print(self.volume)
        print(180/np.pi * self.contact_angle)
        print(self.scale)
        print()

        print('Kapton')
        fit_results = self.fit(model='kapton')
        self.h = fit_results.x[0]
        self.f = fit_results.x[1]
        self.scale = fit_results.x[2]
        print(fit_results)
        print(self.h)
        print(self.f)
        print(self.scale)
        print()

        print('both')
        fit_results = self.fit(model='both')
        self.h = fit_results.x[0]
        self.f = fit_results.x[1]
        self.vf = fit_results.x[2]
        self.volume = fit_results.x[3]
        self.contact_angle = fit_results.x[4]
        self.scale = fit_results.x[5]
        print(fit_results)
        print(self.h)
        print(self.f)
        print(self.vf)
        print(self.volume)
        print(180/np.pi * self.contact_angle)
        print(self.scale)
        print()

        print('Get Absorption')
        params = [
            self.angle,
            self.h,
            self.f,
            self.vf,
            self.volume,
            self.contact_angle
            ]
        self.absorption, self.absorption_water = self.get_absorption(
            model='both',
            s1_norm=setup.s1_norm,
            params=params
            )
        print('Uncertainty')
        self.model_uncertainty(setup)

        params = (
            self.angle, self.h, self.f,
            self.vf, self.volume, self.contact_angle,
            self.scale, self.t
            )
        np.savetxt(
            self.save_to_dir + '/Params_' + str(self.frame).zfill(5) + '.txt',
            params
            )

        if self.plots:
            print('Plotting')
            self.plot_results(setup)
        return None

    def _line_intensity(self, max_abs_x, max_abs_y, angle):
        # This calculates the mean intensity along a line defined by
        # max_abs_x, max_abs_y and angle.
        y = np.arange(self.I.shape[0])
        x = np.tan(-angle)*(y-max_abs_y) + max_abs_x
        x = np.rint(x).astype(int)
        indices = self.mask_inv[y, x]
        pixels = self.normalized_image[y[indices], x[indices]]
        return np.median(pixels)

    def initial_guess(self):
        def lorentzian_skew(x, x0, a, sl, su, m, b):
            bound = np.argmax(x >= x0)
            curve = np.zeros(x.size)
            curve = m*x + b
            curve[:bound] += a / (1 + ((x[:bound] - x0) / sl)**2)
            curve[bound:] += a / (1 + ((x[bound:] - x0) / su)**2)
            return curve

        def lorentzian_skew_jac(x, x0, a, sl, su, m, b):
            peak = np.zeros(x.size)
            dcurve_dx0 = np.zeros(x.size)
            dcurve_dsl = np.zeros(x.size)
            dcurve_dsu = np.zeros(x.size)
            dcurve_dm = np.ones(x.size)
            dcurve_db = np.ones(x.size)

            bound = np.argmax(x >= x0)
            peak[:bound] += a / (1 + ((x[:bound] - x0) / sl)**2)
            peak[bound:] += a / (1 + ((x[bound:] - x0) / su)**2)
            dcurve_dx0[:bound] += \
                2/a * peak[:bound]**2 * ((x[:bound] - x0) / sl**2)
            dcurve_dx0[bound:] += \
                2/a * peak[bound:]**2 * ((x[bound:] - x0) / su**2)
            dcurve_da = peak / a
            dcurve_dsl[:bound] += \
                2/a * peak[:bound]**2 * ((x[:bound] - x0)**2 / sl**3)
            dcurve_dsu[bound:] += \
                2/a * peak[bound:]**2 * ((x[bound:] - x0)**2 / su**3)
            dcurve_dm = x
            jac = np.column_stack((
                dcurve_dx0,
                dcurve_da,
                dcurve_dsl,
                dcurve_dsu,
                dcurve_dm,
                dcurve_db
                ))
            return jac

        # This gets an initial guess for angle, h & f by finding
        # the point of maximum and minimum absorptin
        # Rough location near the point of maximum absorption
        #   max_abs_y - center of detector
        #   max_abs_x - point at maximum absorption
        #   angle - within 20 degrees
        max_abs_y = int(self.I.shape[0] / 2)
        max_abs_x = self.max_abs_point
        dx = 150
        new_angle = self.angle
        d_angle = 20 * np.pi/180

        max_abs_x_array = np.arange(max_abs_x - dx, max_abs_x + dx, 1)
        divisors = [1, 3, 5]
        for d in divisors:
            angle_array = np.arange(
                new_angle - d_angle / d,
                new_angle + d_angle / d + 0.1*np.pi/180,
                0.1*np.pi/180
                )
            medianI_angle = np.zeros(angle_array.size)
            for index, angle in enumerate(angle_array):
                medianI_angle[index] = \
                    self._line_intensity(max_abs_x, max_abs_y, angle)
            # To find the correct angle - a curve is fit to this mean intensity
            # A two-sided lorentzian function is fit to the curve
            popt_median, pcov = curve_fit(
                lorentzian_skew, angle_array, medianI_angle,
                p0=(new_angle, -1, 0.01, 0.01,  0, 1), jac=lorentzian_skew_jac
                )
            y = gaussian_filter1d(medianI_angle, 5)
            angle_filter = angle_array[np.argmin(y)]
            p = np.polyfit(angle_array, medianI_angle, deg=10)
            pfit_curve = np.polyval(p, angle_array)
            angle_pfit = angle_array[np.argmin(pfit_curve)]
            new_angle = (popt_median[0] + angle_pfit + angle_filter)/3
            # Do a similar procedure for the max_abs_x location
            if d in divisors[:-1]:
                meanI_x = np.zeros(max_abs_x_array.size)
                for index, x in enumerate(max_abs_x_array):
                    meanI_x[index] = \
                        self._line_intensity(x, max_abs_y, new_angle)
                y = gaussian_filter1d(medfilt(meanI_x, 21), 5)
                max_abs_x = max_abs_x_array[np.argmin(y)]
        return new_angle

    def fit(self, model):
        params = [self.angle, self.h, self.f, self.vf, self.volume, self.contact_angle]
        if model == 'both':
            x0 = (self.h, self.f, self.vf, self.volume, self.contact_angle, self.scale)
            bounds = (
                (0.001, None),
                (0.01, None),
                (-0.99, 0.99),
                (0.05, None),
                (0, np.pi/2),
                (0, None)
                )
            absorption_temp = None
        elif model == 'kapton':
            x0 = (self.h, self.f, self.scale)
            bounds = (
                (0.001, None),
                (0.01, None),
                (0, None)
                )
            absorption_temp = self.get_absorption(
                'water',
                self.s1_norm_bin,
                params
                )
        elif model == 'water':
            x0 = (self.h, self.vf, self.volume, self.contact_angle, self.scale)
            bounds = (
                (0.001, None),
                (-0.99, 0.99),
                (0.05, None),
                (0, np.pi/2),
                (0, None)
                )
            absorption_temp = self.get_absorption(
                'kapton',
                self.s1_norm_bin,
                params
                )

        fit_results = minimize(
            self._target_function_bin,
            x0=x0,
            method='Nelder-Mead',
            bounds=bounds,
            args=(model, absorption_temp)
            )
        return fit_results

    def _target_function_bin(self, params_in, model, absorption_temp):
        if model == 'both':
            params = [self.angle, params_in[0], params_in[1], params_in[2], params_in[3], params_in[4]]
            scale = params_in[-1]
            absorption, water_absorption = \
                self.get_absorption(model, self.s1_norm_bin, params)
        elif model == 'kapton':
            params = [self.angle, params_in[0], params_in[1], self.vf, self.volume, self.contact_angle]
            scale = params_in[-1]
            water_absorption = absorption_temp
            absorption = self.get_absorption(model, self.s1_norm_bin, params)
        elif model == 'water':
            params = [self.angle, params_in[0], self.f, params_in[1], params_in[2], params_in[3]]
            scale = params_in[-1]
            absorption = absorption_temp
            water_absorption = self.get_absorption(model, self.s1_norm_bin, params)
        absorption_total = absorption * water_absorption
        difference = 1 - self.normalized_image_bin * scale / absorption_total
        return np.linalg.norm(difference)**2

    def param_conversion(self, h, vf, volume, contact_angle):
        c = np.cos(contact_angle)
        r_drop = (3/np.pi * volume) / ((2 + c) * (1 - c)**2)
        offset = r_drop * c
        v = vf * np.sqrt(r_drop**2 - (offset + h)**2)
        return r_drop, offset, v

    def get_absorption(self, model, s1_norm, params):
        def distance(n, s1_norm):
            return np.linalg.norm(n)**2 / np.matmul(s1_norm, n)

        angle, h, f, vf, volume, contact_angle = params
        r_drop, offset, v = self.param_conversion(h, vf, volume, contact_angle)

        n1 = -h * np.array((1, 0, 0)).T
        n2 = -(h + self.t) * np.array((1, 0, 0)).T
        n3 = f * np.array((0, 0, 1)).T
        n4 = -(h + offset) * np.array((1, 0, 0)).T + v * np.array((0, 1, 0)).T

        Rz = np.array((
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1)
            ))
        n1 = np.matmul(Rz, n1)
        n2 = np.matmul(Rz, n2)
        n3 = np.matmul(Rz, n3)
        n4 = np.matmul(Rz, n4)

        L1 = distance(n1, s1_norm)
        if model == 'kapton' or model == 'both' or model == 'angle':
            L2 = distance(n2, s1_norm)
            L3 = distance(n3, s1_norm)

            L2_L3 = L2 >= L3
            indices1 = np.logical_and(L1 < L3, L2_L3)
            indices2 = np.logical_and(np.invert(L2_L3), L2 >= 0)

            path_length = np.zeros(s1_norm.shape[:-1])
            path_length[indices1] = L3[indices1] - L1[indices1]
            path_length[indices2] = L2[indices2] - L1[indices2]

        if model == 'water' or model == 'both' or model == 'angle':
            n4_s1norm = np.matmul(s1_norm, n4)
            n4_mag = np.linalg.norm(n4)
            L4 = n4_s1norm + np.sqrt(n4_s1norm**2 - (n4_mag**2 - r_drop**2))

            indices_water = np.logical_or(L4 < L1, L1 <= 0)
            path_length_water = L1
            path_length_water[indices_water] = L4[indices_water]

        if model == 'kapton' or model == 'both' or model == 'angle':
            absorption = np.exp(-path_length / self.kapton_absorption_length)
        if model == 'water' or model == 'both' or model == 'angle':
            absorption_water = \
                np.exp(-path_length_water / self.water_absorption_length)
        if model == 'kapton':
            return absorption
        elif model == 'water':
            return absorption_water
        elif model == 'both' or model == 'angle':
            return absorption, absorption_water

    def uncertainty_tf(
            self, params, Ic, Ic_theta2_mean, sigma_I_A_2, I_A_2_2, hess
            ):
        s_fac = params[0]
        sigma_A = params[1]
        n = Ic.size
        
        # TF calculation
        sigma_ev11_2 = s_fac**2 * (sigma_I_A_2 + sigma_A**2 * I_A_2_2)
        delta_2 = (n-1)/n * (Ic - Ic_theta2_mean)**2 / sigma_ev11_2
        
        sqrt_delta_2_mean = np.sqrt(delta_2.sum()/n)
        TF = (1 - sqrt_delta_2_mean)**2
        
        # First derivatives
        pTF = (1 - 1/sqrt_delta_2_mean) * 1/n
        pdelta_2 = -delta_2 / sigma_ev11_2

        psigma_ev11_psfac = 2 * sigma_ev11_2 / s_fac
        psigma_ev11_psigma_A = 2 * s_fac**2 * I_A_2_2 * sigma_A

        pdelta_2_psfac = pdelta_2 * psigma_ev11_psfac
        pdelta_2_psigma_A = pdelta_2 * psigma_ev11_psigma_A

        pTF_psfac = pTF * pdelta_2_psfac.sum()
        pTF_psigma_A = pTF * pdelta_2_psigma_A.sum()

        grad = (pTF_psfac, pTF_psigma_A)

        if hess:
            # Second derivatives
            # I don't believe a second derivative based optimization
            # improves performance, but I'm just going to leave
            # them here. Accuracy has been verified
            p2sigma_ev11_p2sfac = 2 * sigma_ev11_2 / s_fac**2
            p2sigma_ev11_p2sigma_A = 2 * s_fac**2 * I_A_2_2
            p2sigma_ev11_psfac_psigma_A = 2 * s_fac * I_A_2_2 * 2 * sigma_A
            
            delta_2_2 = delta_2 / sigma_ev11_2
            delta_2_1 = delta_2_2 / sigma_ev11_2
            p2delta_2_p2sfac = 2 * delta_2_1 * psigma_ev11_psfac**2 \
                - delta_2_2 * p2sigma_ev11_p2sfac
            
            p2delta_2_p2sigma_A = 2 * delta_2_1 * psigma_ev11_psigma_A**2 \
                - delta_2_2 * p2sigma_ev11_p2sigma_A
            
            p2delta_2_psfac_psigma_A = 2 * delta_2_1 * psigma_ev11_psfac * psigma_ev11_psigma_A \
                - delta_2_2 * p2sigma_ev11_psfac_psigma_A
            
            factor1 = (1 - 1/sqrt_delta_2_mean) / n
            factor2 =  1/2 * (delta_2.sum()/n)**(-3/2) / n **2
            p2TF_p2sfac = factor1 * p2delta_2_p2sfac.sum() \
                + factor2 * pdelta_2_psfac.sum()**2
            
            p2TF_p2sigma_A = factor1 * p2delta_2_p2sigma_A.sum() \
                + factor2 * pdelta_2_psigma_A.sum()**2
            
            p2TF_psfac_psigma_A = factor1 * p2delta_2_psfac_psigma_A.sum() \
                + factor2 * pdelta_2_psfac.sum() * pdelta_2_psigma_A.sum()
            
            hess = np.array([
                [p2TF_p2sfac, p2TF_psfac_psigma_A],
                [p2TF_psfac_psigma_A, p2TF_p2sigma_A]
                ])
            return TF, grad, hess
        else:
            return TF, grad

    def uncertainty_fit(
            self, s_fac, sigma_A, Ic, I, Ic_theta2_mean, sigma_I, A, method
            ):
        sigma_I_A_2 = (sigma_I / A)**2
        I_A_2_2 = (I / A**2)**2
        bounds = ((0, None), (0, None))
        if method in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']:
            # This is for second derivative based optimization
            if method == 'Newton-CG':
                bounds = None
            args = (Ic, Ic_theta2_mean, sigma_I_A_2, I_A_2_2, True)
            obj = MemoizeJacHess(self.uncertainty_tf)
            res = scipy.optimize.minimize(
                obj,
                x0=[s_fac, sigma_A],
                bounds=bounds,
                args=args,
                method=method,
                jac=obj.derivative,
                hess=obj.hessian
                )
        else:
            args = (Ic, Ic_theta2_mean, sigma_I_A_2, I_A_2_2, False)
            res = scipy.optimize.minimize(
                self.uncertainty_tf,
                x0=[s_fac, sigma_A],
                bounds=bounds,
                args=args,
                method=method,
                jac=True,
                )
        return res.x[0], res.x[1]

    def _uncertainty_theta2(
            self,
            Ic,
            I,
            sigma_Ic,
            sigma_I,
            I_theta2_variance,
            absorption,
            mask_inv_flat,
            theta2_bounds,
            n_phi_bins_err,
            start_stop_phi,
            method
            ):
        err_array_theta2 = np.zeros((n_phi_bins_err, 4))
        Ic_theta2_mean = Ic[theta2_bounds[0]: theta2_bounds[1]].mean()
        for phi_index in range(n_phi_bins_err):
            phi_start = start_stop_phi[phi_index]
            phi_stop = start_stop_phi[phi_index + 1]
            if phi_stop != 0 and phi_start != 0 and (phi_stop - phi_start) > 0:
                mask_inv_bin = mask_inv_flat[phi_start: phi_stop]
                m = np.count_nonzero(mask_inv_bin)
                if m != 0:
                    Ic_bin = Ic[phi_start: phi_stop][mask_inv_bin]
                    I_bin = I[phi_start: phi_stop][mask_inv_bin]
                    sigma_Ic_bin = np.sqrt(
                        sigma_Ic[phi_start: phi_stop][mask_inv_bin]**2
                        + I_theta2_variance
                        )
                    sigma_I_bin = np.sqrt(
                        sigma_I[phi_start: phi_stop][mask_inv_bin]**2
                        + I_theta2_variance
                        )
                    absorption_bin = \
                        absorption[phi_start: phi_stop][mask_inv_bin]

                    delta = \
                        ((m-1)/m)**(1/2) \
                        * (Ic_bin - Ic_theta2_mean) / sigma_Ic_bin
                    delta.sort()

                    lims = np.array([-1, 1])
                    rankits = scipy.stats.norm.ppf(
                        (np.arange(1, m + 1) - 0.5) / m
                        )
                    fit_indices = np.logical_and(
                        rankits > lims[0], rankits < lims[1]
                        )

                    A = np.vstack((
                        delta[fit_indices],
                        np.ones(np.count_nonzero(fit_indices))
                        )).T
                    s_fac_init, c = np.linalg.lstsq(
                        A, rankits[fit_indices], rcond=None
                        )[0]
                    sigma_A_init = np.abs(c)

                    s_fac, sigma_A = self.uncertainty_fit(
                        s_fac_init, sigma_A_init,
                        Ic_bin, I_bin, Ic_theta2_mean,
                        sigma_I_bin,
                        absorption_bin,
                        method
                        )

                    err_array_theta2[phi_index, 0] = s_fac_init
                    err_array_theta2[phi_index, 1] = sigma_A_init
                    err_array_theta2[phi_index, 2] = s_fac
                    err_array_theta2[phi_index, 3] = sigma_A
        return err_array_theta2

    def model_uncertainty(self, setup, method='L-BFGS-B'):
        I_desampled = np.interp(
            self.theta2_bins_err,
            self.az_average[:, 0],
            self.az_average[:, 1]
            )

        I_theta2_variance = (I_desampled[1:] - I_desampled[:-1])**2 / 12
        mask_inv_flat = self.mask_inv.ravel()[setup.sort_indices]
        I = np.ravel(self.I / setup.polarization)[setup.sort_indices]

        if setup.pedestal_rms is None:
            sigma_I = np.ravel(
                1 / setup.polarization * np.sqrt(self.I)
                )[setup.sort_indices]
        else:
            sigma_I = np.ravel(
                1 / setup.polarization * np.sqrt(self.I + setup.pedestal_rms**2)
                )[setup.sort_indices]

        absorption = np.ravel(
            self.absorption * self.absorption_water
            )[setup.sort_indices]
        Ic = I / absorption
        sigma_Ic = sigma_I / absorption
        
        self.err_array = np.ones(
                (self.n_theta2_bins_err, self.n_phi_bins_err.max(), 4)
                )
        for theta2_index in range(self.n_theta2_bins_err):
            theta2_bounds = [
                setup.start_stop[theta2_index][0],
                setup.start_stop[theta2_index][-1]
                ]
            theta2_good = np.all((
                theta2_bounds[0] != 0,
                theta2_bounds[1] != 0,
                (theta2_bounds[1] - theta2_bounds[0]) > 0
                ))
            if theta2_good:
                self.err_array[theta2_index, :self.n_phi_bins_err[theta2_index], :] = \
                    self._uncertainty_theta2(
                        Ic,
                        I,
                        sigma_Ic,
                        sigma_I,
                        I_theta2_variance[theta2_index],
                        absorption,
                        mask_inv_flat,
                        theta2_bounds,
                        self.n_phi_bins_err[theta2_index],
                        setup.start_stop[theta2_index][:],
                        method
                        )
        np.save(
            self.save_to_dir + '/err_array_' + str(self.frame).zfill(5) + '.npy',
            self.err_array
            )
        np.save(
            self.save_to_dir + '/theta2_bins_err_' + str(self.frame).zfill(5) + '.npy',
            self.theta2_bins_err
            )
        np.save(
            self.save_to_dir + '/phi_bins_err_' + str(self.frame).zfill(5) + '.npy',
            self.phi_bins_err
            )
        return None

    def get_max_intensity(self):
        I = self.I[self.mask_inv]
        upperlimit = I.mean() + 10 * I.std()
        bins_raw_image = np.arange(0, upperlimit, 0.1)
        centers_raw_image = (bins_raw_image[1:] + bins_raw_image[:-1]) / 2
        hist_raw_image = numba_histogram(
            self.I,
            bins=bins_raw_image
            )
        cummulative_sum = (hist_raw_image / hist_raw_image.sum()).cumsum()
        index = np.where(cummulative_sum > 0.9999)[0][0]
        self.max_intensity = centers_raw_image[index]
        if self.plots:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
            axes.plot(centers_raw_image, hist_raw_image / hist_raw_image.sum())
            axes_r = axes.twinx()
            axes_r.plot(centers_raw_image, cummulative_sum, color=[1, 0, 0])
            axes_r.plot([self.max_intensity, self.max_intensity], [0, 1])
            fig.savefig(
                self.save_to_dir
                + '/MaximumIntensity_' + str(self.frame).zfill(5) + '.png'
                )
            plt.close()
        return None

    def plot_results(self, setup):
        self.get_max_intensity()

        ###############
        # Uncertainty #
        ###############
        s_fac_init_array = np.zeros(self.I.shape)
        sigma_A_init_array = np.zeros(self.I.shape)
        s_fac_array = np.zeros(self.I.shape)
        sigma_A_array = np.zeros(self.I.shape)

        labels_theta2 = \
            np.searchsorted(setup.theta2_bins_err, setup.theta2)
        
        labels_phi = np.zeros(labels_theta2.shape)
        for index in range(self.n_theta2_bins_err):
            indices = labels_theta2 == index + 1
            labels_phi_here = np.searchsorted(setup.phi_bins_err[index], setup.phi)
            labels_phi[indices] += labels_phi_here[indices]
        
        labels_all = self.n_phi_bins_err.max() * (labels_theta2 - 1) + (labels_phi - 1)
        indices = np.logical_and(
            labels_all >= 0,
            labels_all < self.n_theta2_bins_err * self.n_phi_bins_err.max()
            )

        labels_index = labels_all[indices].astype(int)
        s_fac_init_array[indices] = \
            self.err_array[:, :, 0].ravel()[labels_index]
        sigma_A_init_array[indices] = \
            self.err_array[:, :, 1].ravel()[labels_index]
        s_fac_array[indices] = \
            self.err_array[:, :, 2].ravel()[labels_index]
        sigma_A_array[indices] = \
            self.err_array[:, :, 3].ravel()[labels_index]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].imshow(s_fac_init_array, vmin=0.75, vmax=1.25)
        axes[0, 1].imshow(sigma_A_init_array, vmin=0, vmax=0.4)

        im0 = axes[1, 0].imshow(s_fac_array, vmin=0.75, vmax=1.25)
        im1 = axes[1, 1].imshow(sigma_A_array, vmin=0, vmax=0.4)

        fig.colorbar(im0, ax=axes[0, 0])
        fig.colorbar(im1, ax=axes[0, 1])

        for row in range(2):
            for column in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_title('s_fac')
        axes[0, 1].set_title('$\sigma_{2\\theta,\phi}^{A}$')
        axes[0, 0].set_ylabel('Initialization')
        axes[1, 0].set_ylabel('Optimized')
        fig.savefig(
            self.save_to_dir
            + '/AbsorptionUncertainty_' + str(self.frame).zfill(5) + '.png'
            )
        plt.cla()
        plt.clf() 
        plt.close('all')

        ########################
        # 2D absorption models #
        ########################
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        im = [[] for index in range(2)]
        im[0] = axes[0].imshow(
            self.absorption,
            vmin=0.5, vmax=1
            )
        im[1] = axes[1].imshow(
            self.absorption_water,
            vmin=0.5, vmax=1
            )
        for column in range(2):
            axes[column].set_xticks([])
            axes[column].set_yticks([])
        fig.colorbar(im[1], ax=axes[1])
        axes[0].set_title('Kapton Absorption')
        axes[1].set_title('Water Absorption')
        fig.tight_layout()
        fig.savefig(
            self.save_to_dir
            + '/Absorption_' + str(self.frame).zfill(5) + '.png'
            )
        plt.cla()
        plt.clf() 
        plt.close('all')

        ########
        # Mask #
        ########
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        im0 = axes[0].imshow(self.I, vmin=0, vmax=self.max_intensity)
        axes[1].imshow(self.mask, vmin=0, vmax=1)
        for column in range(2):
            axes[column].set_xticks([])
            axes[column].set_yticks([])
        axes[0].set_title('Image')
        axes[1].set_title('Mask')
        fig.tight_layout()
        fig.savefig(
            self.save_to_dir + '/Mask_' + str(self.frame).zfill(5) + '.png'
            )
        plt.cla()
        plt.clf() 
        plt.close('all')

        ##########
        # Images #
        ##########
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(self.I, vmin=0, vmax=self.max_intensity)
        im0 = axes[0, 1].imshow(
            self.I / (self.absorption * self.absorption_water),
            vmin=0, vmax=self.max_intensity
            )

        axes[1, 0].imshow(self.normalized_image, vmin=0.95, vmax=1.15)
        im1 = axes[1, 1].imshow(
            self.normalized_image / (self.absorption * self.absorption_water),
            vmin=0.95, vmax=1.15
            )
        for row in range(2):
            for column in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        fig.colorbar(im0, ax=axes[0, 1], fraction=0.05)
        fig.colorbar(im1, ax=axes[1, 1], fraction=0.05)
        axes[0, 0].set_title('Initial')
        axes[0, 1].set_title('Corrected')
        axes[0, 0].set_ylabel('Raw Image')
        axes[1, 0].set_ylabel('Normalized Image')
        fig.tight_layout()
        fig.savefig(
            self.save_to_dir + '/Images_' + str(self.frame).zfill(5) + '.png'
            )
        plt.cla()
        plt.clf() 
        plt.close('all')

        ############
        # 1D plots #
        ############
        D = np.abs(setup.s1[:, :, 2])[self.mask_inv].mean()
        delta = D * (self.h + self.t) / self.f
        alpha = self.angle
        b = delta / np.cos(alpha)
        m = -np.tan(alpha)

        xx, yy = np.meshgrid(
            np.linspace(0, self.I.shape[1]*setup.pixel_size, self.I.shape[1]),
            np.linspace(0, self.I.shape[0]*setup.pixel_size, self.I.shape[0])
            )

        # This assumes the beam center is at x,y = 0,0
        xx += setup._min_pos[0]
        yy += setup._min_pos[1]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        delta_x = (xx - (m * yy - b)) * np.cos(alpha)

        delta_x_int = np.linspace(delta_x.min(), delta_x.max(), 100)
        bin_centers = (delta_x_int[1:] + delta_x_int[:-1]) / 2

        corrected = \
            self.normalized_image / (self.absorption * self.absorption_water)

        averaged_normalized, _ = numba_binned_mean(
            delta_x[self.mask_inv].ravel(),
            bins=delta_x_int,
            weights=self.scale * self.normalized_image[self.mask_inv].ravel()
            )
        averaged_corrected, _ = numba_binned_mean(
            delta_x[self.mask_inv].ravel(),
            bins=delta_x_int,
            weights=corrected[self.mask_inv].ravel()
            )
        averaged_absorption, _ = numba_binned_mean(
            delta_x[self.mask_inv].ravel(),
            bins=delta_x_int,
            weights=self.absorption[self.mask_inv].ravel()
            )
        averaged_absorption_water, _ = numba_binned_mean(
            delta_x[self.mask_inv].ravel(),
            bins=delta_x_int,
            weights=self.absorption_water[self.mask_inv].ravel()
            )

        axes[0, 0].plot(
            bin_centers, averaged_normalized,
            label='Normalized Image'
            )
        axes[0, 0].plot(
            bin_centers, averaged_absorption,
            label='Kapton Absorption'
            )
        axes[0, 0].plot(
            bin_centers, averaged_absorption_water,
            label='Water Absorption'
            )
        axes[0, 0].plot(
            bin_centers, averaged_absorption * averaged_absorption_water,
            label='Combined Absorption'
            )

        axes[1, 0].plot(
            bin_centers, averaged_normalized,
            linewidth=1, label='Normalized Image'
            )
        axes[1, 0].plot(
            bin_centers, averaged_corrected,
            linewidth=1, label='Corrected Image'
            )

        delta_y = (yy + m * xx) * np.cos(alpha)
        delta_y_int = np.linspace(delta_y.min(), delta_y.max(), 40)
        bin_centers = (delta_y_int[1:] + delta_y_int[:-1]) / 2

        averaged_normalized, _ = numba_binned_mean(
            delta_y[self.mask_inv].ravel(),
            bins=delta_y_int,
            weights=self.scale * self.normalized_image[self.mask_inv].ravel()
            )
        averaged_corrected, _ = numba_binned_mean(
            delta_y[self.mask_inv].ravel(),
            bins=delta_y_int,
            weights=corrected[self.mask_inv].ravel()
            )
        averaged_absorption, _ = numba_binned_mean(
            delta_y[self.mask_inv].ravel(),
            bins=delta_y_int,
            weights=self.absorption[self.mask_inv].ravel()
            )
        averaged_absorption_water, _ = numba_binned_mean(
            delta_y[self.mask_inv].ravel(),
            bins=delta_y_int,
            weights=self.absorption_water[self.mask_inv].ravel()
            )

        axes[0, 1].plot(
            bin_centers, averaged_normalized,
            label='Normalized Image'
            )
        axes[0, 1].plot(
            bin_centers, averaged_absorption,
            label='Kapton Absorption'
            )
        axes[0, 1].plot(
            bin_centers, averaged_absorption_water,
            label='Water Absorption'
            )
        axes[0, 1].plot(
            bin_centers, averaged_absorption * averaged_absorption_water,
            label='Combined Absorption'
            )

        axes[1, 1].plot(
            bin_centers, averaged_normalized,
            label='Normalized Image'
            )
        axes[1, 1].plot(
            bin_centers, averaged_corrected,
            linewidth=1, label='Corrected Image'
            )

        axes[1, 0].set_xlabel(
            'Perpendicual distance from\nmaximum absorption (mm)'
            )
        axes[1, 1].set_xlabel(
            'Parallel distance from\nmaximum absorption (mm)'
            )
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].legend(fontsize=10, borderaxespad=0.2)
        axes[1, 0].legend(fontsize=10, borderaxespad=0.2)
        fig.tight_layout()
        fig.savefig(
            self.save_to_dir + '/1DPlots_' + str(self.frame).zfill(5) + '.png'
            )
        plt.cla()
        plt.clf() 
        plt.close('all')
        return None

    def apply_corrections(self, setup, refls):
        s1_norm = flumpy.to_numpy(refls["s1"].each_normalize())
        s1_norm[:, 2] *= -1

        absorption_params = (
            self.angle, self.h, self.f,
            self.vf, self.volume, self.contact_angle
            )
        absorption_water, absorption_kapton = \
            self.get_absorption('both', s1_norm, absorption_params)
        absorption = flex.double(absorption_water * absorption_kapton)

        theta2, phi = setup.get_theta2_phi(s1_norm)
        labels_theta2 = np.searchsorted(self.theta2_bins_err, theta2)
        labels_phi = np.searchsorted(self.phi_bins_err, phi)
        labels_all = self.n_phi_bins_err * (labels_theta2 - 1) + labels_phi
        sigma_A = flex.double(self.err_array[:, :, 3].ravel()[labels_all - 1])

        refls["intensity.sum.variance"] = \
            (refls["intensity.sum.variance"] / absorption**2) \
            + (refls["intensity.sum.value"] / absorption**2 * sigma_A)**2
        refls["intensity.sum.value"] /= absorption
        return refls


class kapton_water_correction:
    def __init__(self, expts, refls, params, logger=None):
        self.expts = expts
        self.refls = refls
        self.params = params
        self.logger = logger
        return None

    def __call__(self):
        self.corrected_reflections = flex.reflection_table()

        s1 = flumpy.to_numpy(self.refls["s1"])
        theta2 = np.arctan(np.sqrt(s1[:, 0]**2 + s1[:, 1]**2) / np.abs(s1[:, 2]))
        self.params['theta2_err_lims'] = [theta2.min(), theta2.max()]
        # Create a setup for each h5 file found within the expts
        image_paths = []
        setups = []
        absorption_models = []
        h5_index = -1
        setup_indices = []
        for index, imageset in enumerate(self.expts.imagesets()):
            path = imageset.paths()[0]
            if path not in image_paths:
                image_paths.append(path)
                setups.append(detector_setup(imageset, self.params))
                h5_index += 1
            setup_indices.append(h5_index)

        # Loop throut each h5 file, then each h5 files frame and perform a
        # correction on each frame
        for h5_index, h5_name in enumerate(image_paths):
            image = dxtbx.load(h5_name)
            absorption_model_image = absorption_model(setups[h5_index])
            for image_index, imageset in enumerate(self.expts.imagesets()):
                if imageset.get_path(0) == h5_name:
                    frame = imageset.indices()[0]
                    if frame == 0:
                        print('Frame: %5d' % (frame))
                        refls = self.refls.select(self.refls['id'] == image_index)
                        if self.params['load_results']:
                            found = absorption_model_image.load_results(frame, self.params)
                        if self.params['load_results'] is False or found is False:
                            data = image.get_raw_data(frame)
                            absorption_model_image.update_frame(
                                data, frame, self.params, refls, setups[h5_index]
                                )
                            absorption_model_image.fit_absorption_model(
                                setups[h5_index]
                                )
                        self.corrected_reflections.extend(
                            absorption_model_image.apply_corrections(
                                setups[h5_index], refls
                                )
                            )

        self.corrected_reflections.as_file(
            self.params['save_to_dir'] + '/corrected_reflections.refl'
            )
        return self.corrected_reflections
