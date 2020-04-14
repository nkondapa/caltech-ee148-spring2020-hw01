import numpy as np
import matplotlib.pyplot as plt


def threshold_convolved_image(convolved_image, mode='mean_std2'):


    convolved_image = np.atleast_3d(convolved_image)
    tci = np.copy(convolved_image)

    for i in range(convolved_image.shape[2]):
        m = np.mean(convolved_image[:, :, i])
        s = np.std(convolved_image[:, :, i])

        if mode == 'mean':
            thr = m
        elif mode == 'mean_std1':
            thr = m + 1 * s
        elif mode == 'mean_std2':
            thr = m + 2 * s
        else:
            thr = m
        tci[convolved_image[:, :, i] < thr, i] = 0

    return tci
