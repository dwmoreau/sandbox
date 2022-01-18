import sys
sys.path.append('/home/david/dials_dev/modules/dxtbx/src')
sys.path.append('/home/david/dials_dev/build/lib')
sys.path.append('/home/david/dials_dev/modules')
from cctbx import factor_kev_angstrom
from dials.algorithms.integration.kapton_correction import get_absorption_correction
from dials.array_family import flex
import dxtbx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class kapton_correction_auto():
    def __init__(self, file_name, params, results_dir):
        self.save_to_dir = results_dir
        self.image = dxtbx.load(file_name)
        self.data = self.image.get_raw_data()
        self.detector = self.image.get_detector()
        self.beam = self.image.get_beam()
        self.wavelength = self.beam.get_wavelength()
        self.n_panels = len(self.detector)
        self.pixel_size = self.detector[0].get_pixel_size()[0]
        self.panel_shape = self.detector[0].get_image_size()
        self.panel_size = self.panel_shape[0] * self.panel_shape[1]
        self.kapton_absorption_length\
            = get_absorption_correction()(self.wavelength)
        self.n_pixels_full = self.n_panels * self.panel_size

        # parameters associated with the:
        #   kapton absorption model
        self.angle = params['angle']
        self.h = params['h']
        self.f = params['f']
        self.t = params['t']

        #   finite pathlength through the sample
        self.f_range = params['f_range']
        self.n_int = params['n_int']

        #   water absorption model
        self.water = params['water']
        self.v = params['v']
        self.r_drop = params['r_drop']

        # parameters associated with the algorithm
        #   pad - distance from edge of detector panels to mask
        #   max_intensity - if a pixel is larger than this, include in the mask
        #   clip - if distance to the right of the beam center is larger than
        #       this, don't use in optimization
        self.pad = params['pad']
        self.polarization_fraction = params['polarization_fraction']
        self.max_intensity = params['max_intensity']
        self.clip = params['clip']
        return None

    def __get_polarization(self, phi, theta2):
        factor = np.cos(2*phi)*np.sin(theta2)**2 / (1+np.cos(theta2)**2)
        return 1 - self.polarization_fraction * factor

    def __azimuthal_average(self, theta2, I, mask=None):
        # Returns the azimuthally averaged image
        # The average is calculated with histograms - very fast implementation
        #   counts == number of pixels in a bin
        #   sum == summed intensity in a bin
        #   sum / counts == average pixel intensity
        theta2_bins = np.linspace(0, 60, 121) * np.pi/180
        theta2_centers = (theta2_bins[1:] + theta2_bins[:-1]) / 2
        if mask is None:
            integration_sum = np.histogram(
                theta2, bins=theta2_bins, weights=I
                )
            integration_counts = np.histogram(theta2, bins=theta2_bins)
        else:
            integration_sum = np.histogram(
                theta2[np.invert(mask)],
                bins=theta2_bins,
                weights=I[np.invert(mask)]
                )
            integration_counts = np.histogram(
                theta2[np.invert(mask)],
                bins=theta2_bins
                )
        integrated = integration_sum[0] / integration_counts[0]
        indices = np.invert(np.isnan(integrated))
        integrated_image = np.interp(
            theta2, theta2_centers[indices], integrated[indices]
            )
        return integrated_image

    def __get_theta2_phi(self, array=False):
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

    def __process_panel(self, data, panel, array=False):
        # This returns information about an individual panel to be fed into
        # a single array to represent the entire image

        # s - vector from crystal to each pixel
        x, y = np.meshgrid(
            np.arange(self.panel_shape[0]),
            np.arange(self.panel_shape[1])
            )
        mm = panel.pixel_to_millimeter(flex.vec2_double(
            flex.double(x.flatten()),
            flex.double(y.flatten())
            ))
        s = panel.get_lab_coord(mm).as_numpy_array()

        # If I am working with 2D arrays - this is just used for visualization
        # purposes
        if array:
            I = data.as_numpy_array()
            s_norm = (s.T / np.linalg.norm(s, axis=1)).T.reshape((*self.panel_shape, 3))
            s_norm[:, :, 2] *= -1
            s = s.reshape((*self.panel_shape, 3))
            return I, s, s_norm
        # If working with flattened arrays
        else:
            I = data.as_numpy_array().flatten()
            mask = np.zeros(self.panel_shape, dtype=np.bool)
            # Mask out the edges of the panel
            mask[:self.pad, :] = True
            mask[-self.pad:, :] = True
            mask[:, :self.pad] = True
            mask[:, -self.pad:] = True
            mask = mask.flatten()

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

            # This does two things
            #   1: flags and removes bad panels. These are flagged for two reasons
            #       a: mean(I) / std(I) is low -
            #       b: mean(I) / std(I) is high -
            #   2: Collapse panels where there is no kapton absorption. These
            #       are flagged by the location - to the right of the beam stop.
            #       Instead of removing them, the panel is averaged and given a
            #       weight in the target function equal to the number of pixels
            #       This is done to speed up the algorithm.
            weights = np.ones(mask.size)
            # Catch the case where the panel is already removed from consideration
            if np.invert(mask).sum() == 0:
                pass
            #elif I[mask].mean() / I[mask].std() < 0.1:
            #    mask = np.ones(mask.shape, dtype=np.bool)
            #elif I[mask].mean() / I[mask].std() > 5:
            #    mask = np.ones(mask.shape, dtype=np.bool)
            elif s[:, 0].mean() > self.clip:
                I[0] = I[np.invert(mask)].mean()
                s[0, :] = s[np.invert(mask), :].mean(axis=0)
                mask = np.ones(self.panel_shape, dtype=np.bool).flatten()
                mask[0] = False
                weights[0] = np.invert(mask).sum()

            s_norm = (s.T / np.linalg.norm(s, axis=1)).T
            s_norm[:, 2] *= -1
            return I, s, s_norm, mask, weights

    def get_image(self):
        # Turns the file into single arrays that are used for the analysis
        self.I = np.zeros(self.n_pixels_full)
        self.s = np.zeros((self.n_pixels_full, 3))
        self.s_norm = np.zeros((self.n_pixels_full, 3))
        self.mask = np.zeros(self.n_pixels_full, dtype=np.bool)
        self.weights = np.zeros(self.n_pixels_full)
        for index in range(self.n_panels):
            I_panel, s_panel, s_norm_panel, mask_panel, weights_panel\
                = self.__process_panel(self.data[index], self.detector[index])
            start = index * self.panel_size
            end = (index + 1) * self.panel_size
            self.I[start: end] = I_panel
            self.s[start: end] = s_panel
            self.s_norm[start: end] = s_norm_panel
            self.mask[start: end] = mask_panel
            self.weights[start: end] = weights_panel
        self.mask = np.logical_or(
            self.mask,
            self.I <= 0
            )
        self.n_pixels = np.invert(self.mask).sum()
        self.I = np.delete(self.I, self.mask)
        self.s = np.delete(self.s, self.mask, axis=0)
        self.s_norm = np.delete(self.s_norm, self.mask, axis=0)
        self.weights = np.delete(self.weights, self.mask)
        self.theta2, self.phi = self.__get_theta2_phi()
        self.polarization = self.__get_polarization(self.phi, self.theta2)
        self.integrated_image = self.__azimuthal_average(
            self.theta2,
            (self.I / self.polarization)
            )
        self.normalized_image\
            = self.I / (self.polarization * self.integrated_image)
        return None

    def get_path_length(self, params=None, array=False):
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

        # Returns the pathlength through the kapton film and water droplet

        # s_norm: unit vector pointing from crystal to each detector pixel
        # n1: Vector pointing from crystal to front face of kapton
        # n2: Vector pointing from crystal to back face of kapton
        # n3: Vector pointing from crystal to far edge of kapton
        n1 = -h * np.array((0, 1, 0)).T
        n2 = -(h + self.t) * np.array((0, 1, 0)).T
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
        L1 = np.divide(np.linalg.norm(n1)**2, np.matmul(s_norm, n1))
        L2 = np.divide(np.linalg.norm(n2)**2, np.matmul(s_norm, n2))
        L3 = np.divide(np.linalg.norm(n3)**2, np.matmul(s_norm, n3))

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
                L3 = np.divide(np.linalg.norm(n3)**2, np.matmul(s_norm, n3))
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

        if self.water:
            # This calculates the path length through a water droplet
            # Assumes that the crystal is in a spherical droplet on the kapton film's
            # front face.
            # The crystal is a distance h from the kapton film along the n1 vector.
            #   This is the crystal height model from above.
            # The crystal is a distance v from the droplet's center along the vector
            #   along the kapton film.
            # delta: Vector point from the center of the water droplet on the kapton
            #   film to the center of the crystal
            # s1 = unit vector from crystal to pixel
            # L4 = pathlength from crystal to water droplet surface along n1
            #   L4*n1 = vector from crystal to water droplet surface
            # delta = vector from center of water droplet to crystal
            #   delta = Rz * [h*(0, 1, 0) + v*(1, 0, 0)]
            # r = vector from center of water droplet on kapton film to water droplet
            #   surface such that r = delta + L4*n1 and |r| = r_drop.
            # To calculate L4, start with the relation
            #   r*r  = (delta + L4*n1)*(delta + L4*n1)
            #   => L4^2 + l4 * 2n1*delta + [|delta|^2 - |r|^2] = 0
            #   => L4 = -delta*n1 + sqrt((delta*n1)^2 -(|delta|^2 - r_drop^2)

            delta = h * np.array((0, 1, 0)).T + v * np.array((1, 0, 0)).T
            delta = np.matmul(Rz, delta)
            delta_snorm = np.matmul(self.s_norm, delta)
            delta_mag = np.linalg.norm(delta)
            L4 = -delta_snorm + np.sqrt(delta_snorm**2 - (delta_mag**2 - r_drop**2))

            # If x-ray path does not go through kapton film, the pathlength is equal to
            # the distance from the crystal to water droplet's surface.
            #   if L1 = 0 or L1 > L4:
            #       pathlength = L4
            # Otherwise, the kapton surface truncates the water pathlength and the
            #   if L1 < L4 and L1 != 0:
            #       pathlength = L1
            indices_water1 = np.logical_and(L1 < L4, L1 >= 0)
            path_length_water = np.zeros(s_norm.shape[:-1])
            path_length_water[indices_water1] = L1[indices_water1]
            return path_length, path_length_water
        else:
            return path_length

    def __absorption_calc(self, L, L_abs):
        return np.exp(-L / L_abs)

    def __target_function(self, params):
        path_length = self.get_path_length(params)
        absorption = self.__absorption_calc(
            path_length, self.kapton_absorption_length
            )
        residuals = self.weights * (self.normalized_image - absorption)
        print(params[0] * 180/np.pi)
        print(params[1])
        print(params[2])
        print(np.linalg.norm(residuals))
        print()
        return np.linalg.norm(residuals)

    def get_absorption(self, array=False):
        if array:
            if self.water:
                self.path_length_array, self.path_length_water_array\
                    = self.get_path_length(array=True)
                self.absorption_water_array = self.__absorption_calc(
                    self.path_length_water, self.water_absorption_length
                    )
            else:
                self.path_length_array = self.get_path_length(array=True)
            self.absorption_array = self.__absorption_calc(
                self.path_length_array, self.kapton_absorption_length
                )
        else:
            if self.water:
                self.path_length, self.path_length_water = self.get_path_length()
                self.absorption_water = self.__absorption_calc(
                    self.path_length_water, self.water_absorption_length
                    )
            else:
                self.path_length = self.get_path_length()
            self.absorption = self.__absorption_calc(
                self.path_length, self.kapton_absorption_length
                )
        return None

    def fit_model(self):
        self.fit_results = minimize(
            self.__target_function,
            x0=(self.angle, self.h, self.f),
            method='L-BFGS-B',
            bounds=((0, 2*np.pi), (0.01, None), (0.2, None)),
            options={'ftol': 1e-09}
            )
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
        self.s_array = -1*np.ones((*max_pos[::-1], 3))
        self.s_norm_array = -1*np.ones((*max_pos[::-1], 3))
        self.mask_array = np.zeros(max_pos[::-1], dtype=np.bool)

        for index in range(self.n_panels):
            origin = np.round(
                (self.detector[index].get_origin()[:2] - min_pos) / self.pixel_size,
                decimals=0
                ).astype(np.int)
            I_panel, s_panel, s_norm_panel = self.__process_panel(
                self.data[index], self.detector[index], array=True
                )
            indices = np.zeros(max_pos[::-1], dtype=np.bool)
            indices[
                origin[1]: origin[1] + self.panel_shape[1],
                origin[0]: origin[0] + self.panel_shape[0]
                ] = True
            self.I_array[indices] = I_panel.flatten()
            self.s_array[indices, :] = s_panel.reshape(
                (self.panel_shape[0]*self.panel_shape[1], 3)
                )
            self.s_norm_array[indices, :] = s_norm_panel.reshape(
                (self.panel_shape[0]*self.panel_shape[1], 3)
                )
        self.mask_array = np.logical_or(
            self.mask_array,
            self.I_array <= 0
            )
        self.theta2_array, self.phi_array = self.__get_theta2_phi(array=True)
        self.theta2_array[self.mask_array] = -1
        self.phi_array[self.mask_array] = -1
        self.polarization_array = self.__get_polarization(
            self.phi_array, self.theta2_array
            )
        self.integrated_image_array = self.__azimuthal_average(
            self.theta2_array,
            (self.I_array / self.polarization_array),
            mask=self.mask_array
            )
        self.normalized_image_array\
            = self.I_array / (self.polarization_array * self.integrated_image_array)
        return None

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
        counts = np.histogram(self.phi[indices], bins=phi_bins)
        p_frac = [0, 0.8, 0.85, 0.9, 0.95, 1.0]
        integrated = np.zeros((len(p_frac), phi_centers.size))
        fig, axes = plt.subplots(1, 1)
        for index, p in enumerate(p_frac):
            self.polarization_fraction = p
            polarization = self.__get_polarization(self.phi, self.theta2)
            normalized_image = self.I / (polarization * self.integrated_image)
            phi_sum = np.histogram(
                self.phi[indices],
                bins=phi_bins,
                weights=normalized_image[indices]
                )
            integrated[index, :] = phi_sum[0] / counts[0]
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
        # x Histogram of raw image and normalized image
        bins_raw_image = np.linspace(0, self.max_intensity, 101)
        hist_raw_image, b = np.histogram(
            self.I_array[np.invert(self.mask_array)],
            bins=bins_raw_image
            )
        centers_raw_image = (bins_raw_image[1:] + bins_raw_image[:-1]) / 2
        raw_image_width = bins_raw_image[1] - bins_raw_image[0]

        bins_norm_image = np.linspace(0, 2, 101)
        centers_norm_image = (bins_norm_image[1:] + bins_norm_image[:-1]) / 2
        norm_image_width = bins_norm_image[1] - bins_norm_image[0]
        hist_norm_image, b = np.histogram(
            self.normalized_image_array[np.invert(self.mask_array)],
            bins=bins_norm_image
            )
        hist_norm_image_corrected, b = np.histogram(
            (self.normalized_image_array / self.absorption_array)[np.invert(self.mask_array)],
            bins=bins_norm_image
            )

        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 6))
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

        # kapton & water absorption models
        if self.water:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            im = [[] for index in range(3)]
            im[0] = axes[0].imshow(self.absorption_array)
            im[1] = axes[1].imshow(self.absorption_water_array)
            im[2] = axes[2].imshow(self.absorption_array * self.absorption_water_array)
            for index in range(3):
                fig.colorbar(im[index], ax=axes[index])
                axes[index].set_xticks([])
                axes[index].set_yticks([])
            axes[0].set_title('Kapton Absorption')
            axes[1].set_title('Water Absorption')
            axes[2].set_title('Total Absorption')
            fig.tight_layout()
            fig.savefig(self.save_to_dir + '/Absorption.png')
            plt.close(fig)
        else:
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
        D = self.detector[0].get_distance()
        delta = D * (self.h + self.t) / self.f
        alpha = -1*(self.angle - 3/2 * np.pi)
        b = delta / np.cos(alpha)
        m = np.tan(alpha)

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
            delta_x[np.invert(self.mask_array)].flatten(),
            bins=delta_x_int,
            weights=self.normalized_image_array[np.invert(self.mask_array)].flatten()
            )
        absorption_sum = np.histogram(
            delta_x[np.invert(self.mask_array)].flatten(),
            bins=delta_x_int,
            weights=self.absorption_array[np.invert(self.mask_array)].flatten()
            )
        counts = np.histogram(
            delta_x[np.invert(self.mask_array)].flatten(),
            bins=delta_x_int
            )
        averaged_absorption = absorption_sum[0] / counts[0]
        averaged_normalized = normalized_sum[0] / counts[0]

        fig, axes = plt.subplots(1, 1)
        axes.plot(bin_centers, averaged_normalized, label='Normalized Image')
        axes.plot(bin_centers, averaged_absorption, label='Absorption Model')
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
        return None
