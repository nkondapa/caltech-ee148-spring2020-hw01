import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as ss


def load_kernel(id, path):
    path = '../data/kernels/kernel=' + str(id) + '.npy'
    kernel = np.load(path)
    kernel = np.array(kernel, dtype=np.float32) / 255

    return kernel


def normalize(matrix, mode='unit'):
    if mode == 'unit':
        matrix = matrix/np.linalg.norm(matrix)
    elif mode == 'sum':
        matrix = matrix/np.sum(matrix)

    return matrix

#
# def filter(img_arr, kernel):
#     kernel = normalize(kernel)
#     flipped_kernel1 = np.flip(kernel[:, :, 0]).transpose()
#     flipped_kernel2 = np.flip(kernel[:, :, 1]).transpose()
#     flipped_kernel3 = np.flip(kernel[:, :, 2]).transpose()
#
#     # normalize
#     # flipped_kernel1 = normalize(flipped_kernel1)
#     # flipped_kernel2 = normalize(flipped_kernel2)
#     # flipped_kernel3 = normalize(flipped_kernel3)
#     flipped_kernel = np.stack([flipped_kernel1, flipped_kernel2, flipped_kernel3], axis=2)
#
#     win_size = int(kernel.shape[0]/2)
#     padded_shape = (img_arr.shape[0] + 2 * win_size, img_arr.shape[1] + 2 * win_size, img_arr.shape[2])
#     img_arr_padded = np.zeros(shape=padded_shape)
#     img_arr_padded[win_size:-win_size, win_size:-win_size, :] = img_arr
#
#     # convolved_image = np.zeros(shape=(img_arr.shape[0], img_arr.shape[1]))
#     convolved_image = np.zeros(shape=img_arr.shape)
#     nf = np.sum(img_arr)
#     for i in range(win_size, img_arr.shape[0]):
#         for j in range(win_size, img_arr.shape[1]):
#             print(i, j)
#             patch = img_arr_padded[(i-5):(i+5), (j-5):(j+5), :]
#
#             # patch = normalize(patch)
#             patch1 = patch[:, :, 0]
#             patch2 = patch[:, :, 1]
#             patch3 = patch[:, :, 2]
#
#             # patch1 = normalize(patch1)
#             # patch2 = normalize(patch2)
#             # patch3 = normalize(patch3)
#
#             convolved_image[i, j, 0] = np.sum(flipped_kernel1 * patch1)
#             convolved_image[i, j, 1] = np.sum(flipped_kernel2 * patch2)
#             convolved_image[i, j, 2] = np.sum(flipped_kernel3 * patch3)
#
#     np.save('../data/test/ci=2', convolved_image)


def dot_product_filter(img_arr, kernel):
    # kernel = normalize(kernel)
    kernels = []
    kernels_norms = []
    for i in range(3):
        flat_kernel = kernel[:, :, i].flatten()/np.linalg.norm(kernel[:, :, i].flatten())
        kernels.append(flat_kernel)

    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr

    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(win_size_left, img_arr.shape[0]):
        for j in range(win_size_left, img_arr.shape[1]):
            print(i, j)
            patch = img_arr_padded[(i - win_size_left):(i + win_size_right + 1),
                    (j - win_size_left):(j + win_size_right + 1), :]
            # patch = normalize(patch)
            for k in range(3):
                patchk = patch[:, :, k].flatten()/np.linalg.norm(patch[:, :, k].flatten())
                convolved_image[i, j, k] = kernels[k] @ patchk

    return convolved_image


def inverse_kernel_filter(img_arr, kernel):
    # kernel = normalize(kernel)
    flipped_kernels = []
    for i in range(3):
        # flipped_kernels.append(np.linalg.pinv(normalize(np.flip(kernel[:, :, 0]).transpose())))
        flipped_kernels.append(normalize(np.linalg.pinv(np.flip(kernel[:, :, 0]).transpose())))
        # flipped_kernels.append(np.flip(kernel[:, :, 0]).transpose())
    flipped_kernel = np.stack(flipped_kernels, axis=2)

    win_size_left = int((kernel.shape[0] - 1)/2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr

    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1), :]            # patch = normalize(patch)
            for k in range(3):
                patchk = patch[:, :, k]
                #patchk = normalize(patch[:, :, k])
                convolved_image[i, j, k] = np.sum(flipped_kernels[k] * patchk)

    return convolved_image



def convolve(kernel, img_array):
    pass