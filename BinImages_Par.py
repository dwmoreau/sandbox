from dxtbx import flumpy
from cctbx import factor_kev_angstrom
from dials.algorithms.integration.kapton_correction import get_absorption_correction
from dials.array_family import flex
import dxtbx
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.signal import medfilt2d
import multiprocessing as mp


def bin_panel_par(I_panel):
    I_filtered = medfilt2d(
        I_panel,
        kernel_size=(5, 5)
        )
    
    z = np.abs((I_panel - I_filtered) / I_panel[pad:-pad, pad:-pad].std())
    I_panel[z > 4] = np.nan
    I_panel[I_panel < 0] = np.nan
    nan_indices = np.argwhere(np.isnan(I_panel))
    for nan_index in nan_indices:
        I_panel[
            nan_index[0] - 2: nan_index[0] + 2,
            nan_index[1] - 2: nan_index[1] + 2
            ] = np.nan
    I_panel = np.column_stack((np.zeros(I_panel.shape[0]), I_panel, np.zeros(I_panel.shape[0])))
    I_panel = np.row_stack((np.zeros(I_panel.shape[1]), I_panel, np.zeros(I_panel.shape[1])))

    I_panel[:pad + 1, :] = np.nan
    I_panel[-pad - 1:, :] = np.nan
    I_panel[:, :pad + 1] = np.nan
    I_panel[:, -pad - 1:] = np.nan

    I_reshaped = I_panel.reshape((
        int(I_panel.shape[0] / binning),
        binning,
        int(I_panel.shape[1] / binning),
        binning
        ))
    I_binned = np.nanmedian(np.nanmedian(I_reshaped, axis=3), axis=1)
    if np.isnan(I_binned).sum() > 0:
        try:
            x = np.arange(I_binned.shape[1])
            for row in range(I_binned.shape[0]):
                if np.isnan(I_binned[row, :]).sum() > 0:
                    nan_indices = np.isnan(I_binned[row, :])
                    I_binned[row, nan_indices] = np.interp(
                        x[nan_indices],
                        x[np.invert(nan_indices)],
                        I_binned[row, np.invert(nan_indices)]
                        )
        except:
            nan_indices = np.isnan(I_binned)
            I_binned[nan_indices] = I_binned[np.invert(nan_indices)].mean()
    return I_binned


pad = 2
binning = 32
n_proc = 16

save_to_name = 'absorption_frames_binning_' + str(binning) + '.npy'
file_name = '/net/dials/raid1/sauter/bernina/spectrum_masters/run_000795.JF07T32V01_master.h5'
refls_template = '/net/dials/raid1/swissfel/p18163/res/Cyt/py3/proc/t3/idx-run_000795.JF07T32V01_master_!!!!!_strong.refl'

n_frames = 1448
bad_panels = [
    1, 5, 16, 20, 40, 41, 42, 43, 44, 45, 46, 47, 176, 143, 180, 181,
    193, 196, 197,
    224, 225, 226, 227, 228, 229, 230, 231, 238, 239, 250, 251, 254, 255,
    ]

image = dxtbx.load(file_name)
detector = image.get_detector()
beam = image.get_beam()
wavelength = beam.get_wavelength()
energy = factor_kev_angstrom / wavelength

pixel_size = detector[0].get_pixel_size()[0]


panel_arrangement = np.array([16, 16])
n_panels = panel_arrangement[0] * panel_arrangement[1]

panel_shape = detector[0].get_image_size()
rng = np.random.default_rng()
random_indices = rng.permutation(panel_shape[0]*panel_shape[1])
unrandom_indices = np.argsort(random_indices)

binned_panel_shape = [
    int((panel_shape[0] + 2) / binning),
    int((panel_shape[1] + 2) / binning)
    ]

image_shape = panel_arrangement * binned_panel_shape

min_pos = np.zeros(2)
max_pos = np.zeros(2)
for index in range(n_panels):
    origin = detector[index].get_origin()
    if origin[0] < min_pos[0]:
        min_pos[0] = origin[0]
    if origin[1] < min_pos[1]:
        min_pos[1] = origin[1]
    if origin[0] > max_pos[0]:
        max_pos[0] = origin[0]
    if origin[1] > max_pos[1]:
        max_pos[1] = origin[1]

max_pos[0] += panel_shape[0] * pixel_size
max_pos[1] += panel_shape[1] * pixel_size

rows = np.zeros(n_panels, dtype=int)
columns = np.zeros(n_panels, dtype=int)
for panel_index in range(n_panels):
    origin = detector[panel_index].get_origin()
    panel_row = np.rint(
        panel_arrangement[0] * (origin[0] - min_pos[0]) / (max_pos[0] - min_pos[0])
        ).astype(int)
    panel_column = np.rint(
        panel_arrangement[1] * (origin[1] - min_pos[1]) / (max_pos[1] - min_pos[1])
        ).astype(int)
    rows[panel_index] = panel_row * binned_panel_shape[0]
    columns[panel_index] = panel_column * binned_panel_shape[1]

I = np.zeros((*image_shape, n_frames))

with mp.Pool(n_proc) as p:
    for frame in range(n_frames):
        print(frame)
        data = image.get_raw_data(frame)
        inputs = []
        row = []
        column = []
        for panel_index in range(n_panels):
            if panel_index not in bad_panels:
                row.append(rows[panel_index])
                column.append(columns[panel_index])
                data_panel = flumpy.to_numpy(data[panel_index]) / energy
                inputs.append(data_panel)

        outputs = p.map(bin_panel_par, inputs)
        for index, output in enumerate(outputs):
            I[
                column[index]: column[index] + binned_panel_shape[1],
                row[index]: row[index] + binned_panel_shape[0],
                frame
                ] = output
            
np.save(save_to_name, I)