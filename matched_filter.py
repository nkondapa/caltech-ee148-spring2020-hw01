import numpy as np


def filter_3d(img_arr, kernel):

    kernel = 1/np.std(kernel) * (kernel - np.mean(kernel))

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr

    # generate "convolved" image
    convolved_image = np.zeros(shape=(img_arr.shape[0], img_arr.shape[1]))
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1), :]

            npatch = 1/np.std(patch) * (patch - np.mean(patch))
            convolved_image[i, j] = np.mean(npatch * kernel)

    return convolved_image


def filter_2d(img_arr, kernel):

    if len(img_arr.shape) == 2:
        return filter_2d_one_channel(img_arr, kernel)

    kernels = []
    for i in range(3):
        nkernel = 1/np.std(kernel[:, :, i]) * (kernel[:, :, i] - np.mean(kernel[:, :, i]))
        kernels.append(nkernel)

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr

    # generate "convolved" image
    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1), :]

            for k in range(3):
                pk = patch[:, :, k]
                npk = 1/np.std(pk) * (pk - np.mean(pk))
                convolved_image[i, j, k] = np.mean(npk * kernels[k])

    return convolved_image


def filter_2d_one_channel(img_arr, kernel):
    print('one_channel')
    kernel = np.mean(kernel, axis=2)
    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size)
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right] = img_arr

    # generate "convolved" image
    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1)]

            if np.max(patch) != 0:
                # denom_matrix = np.std(patch)
                # npk = np.divide(1, denom_matrix, out=np.zeros_like(img_arr), where=denom_matrix != 0)
                npk = 1 / np.std(patch) * (patch - np.mean(patch))
                convolved_image[i, j] = np.mean(npk * kernel)

    return convolved_image


def filter_1d(img_arr, kernel):

    kernels = []
    for i in range(3):
        flat_kernel = kernel[:, :, i].flatten()/np.linalg.norm(kernel[:, :, i].flatten())
        kernels.append(flat_kernel)

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size, img_arr.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr

    # generate "convolved" image
    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1), :]
            for k in range(3):
                patchk = patch[:, :, k].flatten()/np.linalg.norm(patch[:, :, k].flatten())
                convolved_image[i, j, k] = kernels[k] @ patchk

    return convolved_image


def smooth(img_arr, kernel):

    # kernel = np.mean(kernel, axis=2)
    # flat_kernel = kernel[:, :].flatten()/np.linalg.norm(kernel[:, :].flatten())

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size)
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right] = img_arr

    # generate "convolved" image
    convolved_image = np.zeros(shape=img_arr.shape)
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            # print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left
            patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                    (pj - win_size_left):(pj + win_size_right + 1)]

            if np.max(patch) == 0:
                continue
            # patchk = patch[:, :].flatten()/np.linalg.norm(patch[:, :].flatten())
            # convolved_image[i, j] = flat_kernel @ patchk
            convolved_image[i, j] = np.sum(patch * kernel)

    return convolved_image
