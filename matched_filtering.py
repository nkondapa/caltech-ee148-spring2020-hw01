import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss


def load_kernel(id, path):
    path = '../data/kernels/kernel=' + str(id) + '.npy'
    kernel = np.load(path)

    return kernel


def normalize(matrix, mode='sum'):
    if mode == 'unit':
        matrix = matrix/np.linalg.norm(matrix)
    elif mode == 'sum':
        matrix = matrix/np.sum(matrix)

    return matrix


def filter(img_arr, kernel):
    # kernel = normalize(kernel)
    flipped_kernel1 = np.flip(kernel[:, :, 0]).transpose()
    flipped_kernel2 = np.flip(kernel[:, :, 1]).transpose()
    flipped_kernel3 = np.flip(kernel[:, :, 2]).transpose()

    # normalize
    flipped_kernel1 = normalize(flipped_kernel1)
    flipped_kernel2 = normalize(flipped_kernel2)
    flipped_kernel3 = normalize(flipped_kernel3)
    flipped_kernel = np.stack([flipped_kernel1, flipped_kernel2, flipped_kernel3], axis=2)

    win_size = int(kernel.shape[0]/2)
    padded_shape = (img_arr.shape[0] + 2 * win_size, img_arr.shape[1] + 2 * win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size:-win_size, win_size:-win_size, :] = img_arr

    # convolved_image = np.zeros(shape=(img_arr.shape[0], img_arr.shape[1]))
    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(win_size, img_arr.shape[0]):
        for j in range(win_size, img_arr.shape[1]):
            patch = img_arr_padded[(i-5):(i+5), (j-5):(j+5), :]

            patch1 = patch[:, :, 0]
            patch2 = patch[:, :, 1]
            patch3 = patch[:, :, 2]

            patch1 = normalize(patch1)
            patch2 = normalize(patch2)
            patch3 = normalize(patch3)

            convolved_image[i, j, 0] = np.sum(flipped_kernel1 * patch1)
            convolved_image[i, j, 1] = np.sum(flipped_kernel2 * patch2)
            convolved_image[i, j, 2] = np.sum(flipped_kernel3 * patch3)

    plt.imshow(convolved_image)
    plt.show()

    # x = 71
    # y = 187
    # x = 319
    # y = 60
    # patch = img_arr[(y-20):(y+20), (x-20):(x+20), :]
    # # patch = normalize(patch)
    # fkernel = np.flip(kernel[:, :, 0]).transpose()
    # ss.convolve2d(kernel[:, :, 0], patch[:, :, 0])
    # # plt.figure()
    # # plt.subplot(2, 2, 1)
    # # plt.imshow(kernel[:, :, 0])
    # # plt.subplot(2, 2, 2)
    # # plt.imshow(fkernel)
    # # plt.subplot(2, 2, 3)
    # # plt.imshow(kernel[:, :, 0] * patch[:, :, 0])
    # # plt.subplot(2, 2, 4)
    # # plt.imshow(fkernel * patch[:, :, 0])
    # # plt.show()
    # print()

def convolve(kernel, patch):
    pass