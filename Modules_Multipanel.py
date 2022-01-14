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


class get_absorption_correction:
    # This is taken from Iris's code
    # The attenuation length is taken from the Center for X-ray Optics at LBL
    #   https://henke.lbl.gov/optical_constants/
    #   https://henke.lbl.gov/optical_constants/atten2.html
    def __init__(self):
        # Kapton, or polyimide.  C22H10N2O5 Density=1.43, Angle=90.deg
        # Photon Energy (eV), Atten Length (microns)
        data = """6000.00  482.643
   6070.00  500.286
   6140.00  518.362
   6210.00  536.896
   6280.00  555.873
   6350.00  575.302
   6420.00  595.191
   6490.00  615.552
   6560.00  636.382
   6630.00  657.691
   6700.00  679.484
   6770.00  701.758
   6840.00  724.521
   6910.00  747.791
   6980.00  771.561
   7050.00  795.846
   7120.00  820.646
   7190.00  845.963
   7260.00  871.812
   7330.00  898.183
   7400.00  925.082
   7470.00  952.535
   7540.00  980.535
   7610.00  1009.09
   7680.00  1038.18
   7750.00  1067.85
   7820.00  1098.08
   7890.00  1128.88
   7960.00  1160.25
   8030.00  1192.20
   8100.00  1224.76
   8170.00  1257.91
   8240.00  1291.67
   8310.00  1326.01
   8380.00  1360.98
   8450.00  1396.54
   8520.00  1432.72
   8590.00  1469.51
   8660.00  1506.93
   8730.00  1544.96
   8800.00  1583.65
   8870.00  1622.95
   8940.00  1662.90
   9010.00  1703.49
   9080.00  1744.72
   9150.00  1786.59
   9220.00  1829.13
   9290.00  1872.31
   9360.00  1916.16
   9430.00  1960.65
   9500.00  2005.82
   9570.00  2051.65
   9640.00  2098.16
   9710.00  2145.36
   9780.00  2193.22
   9850.00  2241.75
   9920.00  2290.95
   9990.00  2340.86
   10060.0  2391.49
   10130.0  2442.84
   10200.0  2494.86
   10270.0  2547.59
   10340.0  2601.02
   10410.0  2655.14
   10480.0  2709.98
   10550.0  2765.49
   10620.0  2821.73
   10690.0  2878.68
   10760.0  2936.31
   10830.0  2994.67
   10900.0  3053.72
   10970.0  3113.49
   11040.0  3173.96
   11110.0  3235.14
   11180.0  3297.07
   11250.0  3359.67
   11320.0  3423.01
   11390.0  3487.04
   11460.0  3551.76
   11530.0  3617.23
   11600.0  3683.38
   11670.0  3750.23
   11740.0  3817.81
   11810.0  3886.07
   11880.0  3955.05
   11950.0  4024.75
   12020.0  4095.11
   12090.0  4166.20
   12160.0  4237.96
   12230.0  4310.40
   12300.0  4383.60
   12370.0  4457.48
   12440.0  4532.02
   12510.0  4607.25
   12580.0  4683.14
   12650.0  4759.73
   12720.0  4837.01
   12790.0  4914.94
   12860.0  4993.54
   12930.0  5072.79
   13000.0  5152.69"""
        self.energy = flex.double()
        self.microns = flex.double()
        for line in data.split("\n"):
            tokens = line.strip().split()
            self.energy.append(float(tokens[0]))
            self.microns.append(float(tokens[1]))

    def __call__(self, wavelength_ang):
        # calculate energy in eV 12398.425 eV/Ang
        energy_eV = 12398.425 / wavelength_ang
        # interpolate the Henke tables downloaded from lbl.gov
        index_float = (
            (len(self.energy) - 1)
            * (energy_eV - self.energy[0])
            / (self.energy[-1] - self.energy[0])
        )
        fraction, int_idx = math.modf(index_float)
        int_idx = int(int_idx)

        microns = self.microns[int_idx] + fraction * (
            self.microns[int_idx + 1] - self.microns[int_idx]
        )
        return microns / 1000.0


def GetModel(angle, h, f, t, s_norm,  v=0, r_drop=0.15, out='kapton'):
    # Returns the pathlength through the kapton film

    # s_norm: unit vector pointing from crystal to each detector pixel
    # n1: Vector pointing from crystal to front face of kapton
    # n2: Vector pointing from crystal to back face of kapton
    # n3: Vector pointing from crystal to far edge of kapton
    # delta: Vector point from the center of the water droplet on the kapton
    #   film to the center of the crystal
    n1 = -h * np.array((0, 1, 0)).T 
    n2 = -(h + t) * np.array((0, 1, 0)).T
    n3 = f * np.array((0, 0, 1)).T
    #delta = h * np.array((0, 1, 0)).T + v * np.array((1, 0, 0)).T

    # Rotate these vectors to account for the kapton angle
    Rz = np.array((
        (np.cos(angle), -np.sin(angle), 0),
        (np.sin(angle), np.cos(angle), 0),
        (0, 0, 1)
        ))

    n1 = np.matmul(Rz, n1)
    n2 = np.matmul(Rz, n2)
    n3 = np.matmul(Rz, n3)
    #delta = np.matmul(Rz, delta)

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
    path_length = np.zeros(s_norm.shape[:-1])

    # For paths through the front face and the far edge
    path_length[indices1] = L3[indices1] - L1[indices1]
    # For paths through the front face and the back face
    path_length[indices2] = L2[indices2] - L1[indices2]

    # This calculates the path length through a water droplet
    # Assumes that the crystal is in a spherical droplet on the kapton film's
    # front face.
    # The crystal is a distance h from the kapton film along the n1 vector.
    #   This is the crystal height model from above.
    # The crystal is a distance v from the droplet's center along the vector
    #   along the kapton film.

    # n1 = unit vector from crystal to pixel
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
    
    #delta_snorm = np.matmul(s_norm, delta)
    #delta_mag = np.linalg.norm(delta)
    #L4 = -delta_snorm + np.sqrt(delta_snorm**2 - (delta_mag**2 - r_drop**2))

    # If x-ray path does not go through kapton film, the pathlength is equal to
    # the distance from the crystal to water droplet's surface.
    #   if L1 = 0 or L1 > L4:
    #       pathlength = L4
    # Otherwise, the kapton surface truncates the water pathlength and the 
    #   if L1 < L4 and L1 != 0:
    #       pathlength = L1
    #indices_water1 = np.logical_and(
    #    L1 < L4,
    #    L1 >= 0
    #    )
    #path_length_water = L4.copy()
    #path_length_water[indices_water1] = L1[indices_water1]
    
    if out == 'kapton':
        return path_length
    elif out == 'water':
        return path_length_water
    elif out == 'both':
        return path_length, path_length_water


def NormalizeImage(data, s, mask, polarization_fraction):
    # Flattens the image - correcting for polarization and radially symmetric 
    # scattering
    # Set up for azimuthal averaging
    # This assumes that the beam direction is (0, 0, -1)
    R = np.sqrt(s[:, 0]**2 + s[:, 1]**2)
    theta2_array = np.arctan(R / np.abs(s[:, 2]))
    theta2_int = np.linspace(3.5, 65, 100) * np.pi/180

    # Polarization correction
    # For more information see the following papers:
    #   Kahn, R. et al. (1982), J. Appl. Cryst. 15, 330-337
    #   Sulyanov, S. et al. (2014), J. Appl. Cryst. 47, 1449-1451
    # polarization fraction is the fraction of x-rays polarized in the
    # orientation of the monochromator crystal
    phi = np.pi - 1*np.arctan2(s[:, 0], s[:, 1])
    polarization = (1 - polarization_fraction*np.cos(2*phi)*np.sin(theta2_array)**2 / (1+np.cos(theta2_array)**2))
    data_polarization_corrected = data / polarization.reshape(data.shape)
    
    #fig, axes = plt.subplots(1, 1)
    #axes.imshow(phi.reshape(data.shape))
    #plt.show()

    # The image quadrant with phi <= pi is unaffected by kapton absorption
    # and odd, unexplained high scattering intensity. This should not be
    # used in the final version of the code
    #phi_indices = np.logical_or(
    #   phi <= np.pi / 2,
    #   phi >= 3 * np.pi / 2 
    #   )
    phi_indices = phi <= np.pi

    # These are the portions of the detector that are used to do the 
    # azimuthal average
    integratation_indices = np.logical_or(
        phi_indices.reshape(data.shape),
        np.invert(mask)
        )
    
    #integratation_indices = np.invert(mask)
    # This calculates the average scatter into a pixel at a given 2theta angle 
    # It works by counting the total number of photons scattered into all the 
    # pixels within a 2theta bin. Then dividing by the total number of pixels
    # in the 2theta bin
    integration_sum = np.histogram(
        theta2_array[integratation_indices.flatten()],
        bins=theta2_int,
        weights=data[integratation_indices].flatten()
        )
    integration_counts = np.histogram(
        theta2_array[integratation_indices.flatten()],
        bins=theta2_int
        )
    
    bins = integration_counts[1]
    bin_centers = (bins[1:] + bins[:-1])/2
    integrated = integration_sum[0] / integration_counts[0]

    # bins with zero pixels produce a divide by zero error resulting in nan
    indices = np.invert(np.isnan(integrated))

    # This expands the 1D azimuthal average to the 2D detector image
    #interpolated = np.interp(bin_centers, bin_centers[indices], integrated[indices])
    integrated_image = np.interp(theta2_array, bin_centers[indices], integrated[indices])

    normalized_image = data_polarization_corrected / integrated_image.reshape(data.shape)
    return normalized_image


def GetAbsorption(path_length, absorption_length):
    return np.exp(-path_length / absorption_length)


def IntegrateModel(angle, h, f, t, s_norm, kapton_absorption_length, f_range, n=1):
    # Returns the absorption from the kapton tape
    # If n == 1, calculate the absorption assuming all the scattering comes 
    #   from a point
    # Else, calculates the absorption assuming a finite pathlength
    #   This is done by absorption = 1/L x integrate(absorption)
    #   If n == 3, the returned absorption is the average of the absorption model
    #   at the start, middle, and end of the pathlength through the water droplet
    #   n == 3 is a good choice - don't see any benefit on n > 5

    if n == 1:
        path_length = GetModel(angle, h, f, t, s_norm, out='kapton')
    else:
        path_length_int = np.zeros((*s_norm.shape[:-1], n))
        f_int = np.linspace(f - f_range, f + f_range, n)
        df = f_int[1] - f_int[0]
        for index, f_here in enumerate(f_int):
            path_length_here = GetModel(angle, h, f_here, t, s_norm, out='kapton')
            if len(path_length_int.shape) == 3:
                path_length_int[:, :, index] = path_length_here
            elif len(path_length_int.shape) == 2:
                path_length_int[:, index] = path_length_here
        path_length = np.trapz(path_length_int, f_int, axis=-1) / (f_int[-1] - f_int[0])
    return GetAbsorption(path_length, kapton_absorption_length)


def WaterAbsorption(angle, h, f, t, s_norm, v, r_drop):
    # Returns the absorption from water
    path_length_water = GetModel(angle, h, f, t, s_norm, v=v, r_drop=r_drop, out='water')
    water_absorption_length = 1.7 # at 9.5 kev or 1.3 A
    return GetAbsorption(path_length_water, water_absorption_length)


def ProcessPanel(data, panel):
    I = data.as_numpy_array()
    panel_shape = panel.get_image_size()
    x, y = np.meshgrid(
        np.linspace(0, panel_shape[0] - 1, panel_shape[0]),
        np.linspace(0, panel_shape[1] - 1, panel_shape[1])
        )
    mm = panel.pixel_to_millimeter(flex.vec2_double(
        flex.double(x.flatten()),
        flex.double(y.flatten())
        ))
    s = panel.get_lab_coord(mm).as_numpy_array()
    s_norm = (s.T / np.linalg.norm(s, axis=1)).T.reshape((panel_shape[0], panel_shape[1], 3))
    #s_norm[:, :, 1] *= -1
    s_norm[:, :, 2] *= -1
    s = s.reshape((panel_shape[0], panel_shape[1], 3))
    return I, s, s_norm


def GetImage(data, detector, polarization_fraction):
    pixel_size = detector[0].get_pixel_size()[0]
    min_pos = np.zeros(2)
    max_pos = np.zeros(2, dtype=np.int)
    for index in range(16*16):
        origin = detector[index].get_origin()
        if origin[0] < min_pos[0]:
            min_pos[0] = origin[0]
        if origin[1] < min_pos[1]:
            min_pos[1] = origin[1]
    for index in range(16*16):
        origin = np.round(
            (detector[index].get_origin()[:2] - min_pos) / pixel_size,
            decimals=0
            ).astype(np.int)
        if origin[0] > max_pos[0]:
            max_pos[0] = origin[0]
        if origin[1] > max_pos[1]:
            max_pos[1] = origin[1]
    max_pos += 254
    I = -1*np.ones(max_pos[::-1])
    s = -1*np.ones((*max_pos[::-1], 3))
    s_norm = -1*np.ones((*max_pos[::-1], 3))
    mask = np.zeros(max_pos[::-1], dtype=np.bool)
    for index in range(16*16):
        origin = np.round(
            (detector[index].get_origin()[:2] - min_pos) / pixel_size,
            decimals=0
            ).astype(np.int)
        I_panel, s_panel, s_norm_panel = ProcessPanel(data[index], detector[index])
        I[origin[1]: origin[1] + 254, origin[0]: origin[0] + 254] = I_panel
        s[origin[1]: origin[1] + 254, origin[0]: origin[0] + 254] = s_panel
        s_norm[origin[1]: origin[1] + 254, origin[0]: origin[0] + 254] = s_norm_panel
    mask = np.logical_or(
        mask,
        I <= 0
        )
    R = np.sqrt(s[:, :, 0]**2 + s[:, :, 1]**2)
    theta2 = np.arctan(R / np.abs(s[:, :, 2]))
    phi = np.pi + np.arctan2(s[:, :, 1], s[:, :, 0])
    polarization = (1 - polarization_fraction*np.cos(2*phi)*np.sin(theta2)**2 / (1+np.cos(theta2)**2))
    theta2_int = np.linspace(0, 60, 121) * np.pi/180
    integratation_indices = np.invert(mask)

    integration_sum = np.histogram(
        theta2[integratation_indices].flatten(),
        bins=theta2_int,
        weights=I[integratation_indices].flatten()
        )
    integration_counts = np.histogram(
        theta2[integratation_indices].flatten(),
        bins=theta2_int
        )
    bins = integration_counts[1]
    bin_centers = (bins[1:] + bins[:-1])/2
    integrated = integration_sum[0] / integration_counts[0]
    indices = np.invert(np.isnan(integrated))
    integrated_image = np.interp(theta2, bin_centers[indices], integrated[indices])
    integrated_image[mask] = -1
    return I, s, s_norm, mask, theta2, phi, polarization, integrated_image


def Gradient(s, normalized_image, mask):
    phi = np.pi - 1*np.arctan2(s[:, 0], s[:, 1])
    phi_indices = np.logical_or(
        phi <= np.pi / 2,
        phi >= 3 * np.pi / 2 
        )
    fit_indices = np.logical_and(
        phi_indices,
        np.invert(mask).flatten()
        )
    x, y = np.meshgrid(
        np.arange(mask.shape[0]),
        np.arange(mask.shape[1])
        )
    A = np.vstack((
        x.flatten(),
        y.flatten(),
        np.ones(mask.shape).flatten()
        )).T
    plane_results = np.linalg.lstsq(
        A[fit_indices, :],
        normalized_image.flatten()[fit_indices],
        rcond=None
        )
    Y = np.matmul(A, plane_results[0]).reshape(mask.shape)
    return Y