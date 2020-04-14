import numpy as np
import matplotlib.pyplot as plt
import time


def pixel_rgb_distance_map(img_arr, kernel, points=None):

    sub_pooled_scores_list = []
    pooled_scores_list = []
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if i != j:
                continue

            kpix = kernel[i, j, :]
            norm_matrix = np.expand_dims(np.linalg.norm(img_arr, axis=2), axis=2)
            norm_img_arr = np.divide(img_arr, norm_matrix, out=np.zeros_like(img_arr), where=norm_matrix != 0)
            scores = 1 - np.linalg.norm(kpix/np.linalg.norm(kpix) - norm_img_arr, axis=2)
            pooled_image = sub_scoring_algs(scores, kernel, (i, j), algs=['pooled'], points=points)[0]
            # sub_pooled_scores_list.append(pooled_image)

    # sub_pooled_scores_arr = np.stack(sub_pooled_scores_list, axis=3)
    return pooled_image


def sub_scoring_algs(img_arr, kernel, kernel_pos, algs=None, points=None):

    if algs is None:
        algs = ['pooled']

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr.shape[0] + total_win_size, img_arr.shape[1] + total_win_size)
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right] = img_arr

    # generate "pooled" image
    pooled_image = np.zeros(shape=img_arr.shape)
    mean_pooled_image = np.zeros(shape=img_arr.shape)
    mapped_image = np.zeros(shape=img_arr.shape)
    sub_pooled_image = np.zeros(shape=img_arr.shape)
    sub_ps = 2

    count = 0
    for i in range(0, img_arr.shape[0], 2):
        for j in range(0, img_arr.shape[1]):
            count += 1

            if points is not None:
                if (i, j) not in points:
                    continue
                else:
                    print(i, j)
            # print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left

            if 'pooled' in algs:
                patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                        (pj - win_size_left):(pj + win_size_right + 1)]
                pooled_image[i, j] = np.max(patch, axis=(0, 1))

            if 'sub_pooled' in algs:
                pikp = pi - (7 - kernel_pos[0])
                pjkp = pj - (7 - kernel_pos[1])
                if pikp - sub_ps < 0 or pjkp - sub_ps < 0:
                    sub_patch = np.zeros(shape=(1, img_arr.shape[2]))
                else:
                    sub_patch = img_arr_padded[(pikp - sub_ps):(pikp + sub_ps + 1),
                            (pjkp - sub_ps):(pjkp + sub_ps + 1)]

                sub_pooled_image[i, j] = np.max(sub_patch, axis=(0, 1))

            if 'mean_pooled' in algs:
                patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                        (pj - win_size_left):(pj + win_size_right + 1)]
                mean_pooled_image[i, j] = np.mean(patch, axis=(0, 1))

    rt = []
    if 'pooled' in algs:
        rt.append(pooled_image)

    if 'mean_pooled' in algs:
        rt.append(mean_pooled_image)

    if 'sub_pooled' in algs:
        rt.append(sub_pooled_image)

    return rt


def pixel_rgb_distance_map_multi_kernel(img_arr, kernel_stack):

    sub_pooled_scores_list = []
    pooled_scores_list = []
    for i in range(kernel_stack.shape[0]):
        for j in range(kernel_stack.shape[1]):
            if i != j:
                continue
            kpixes = kernel_stack[i, j, :, :]
            scores_list = []
            for k in range(kpixes.shape[1]):
                kpix = kpixes[:, k]
                norm_matrix = np.expand_dims(np.linalg.norm(img_arr, axis=2), axis=2)
                norm_img_arr = np.divide(img_arr, norm_matrix, out=np.zeros_like(img_arr), where=norm_matrix != 0)
                scores = 1 - np.linalg.norm(kpix/np.linalg.norm(kpix) - norm_img_arr, axis=2)
                scores_list.append(scores)
            scores_stack = np.stack(scores_list, axis=2)
            pooled_image = sub_scoring_algs_stack(scores_stack, kernel_stack, (i, j), algs=['pooled'])[0]
            pooled_scores_list.append(pooled_image)
            # sub_pooled_scores_list.append(pooled_image)

    # sub_pooled_scores_arr = np.stack(sub_pooled_scores_list, axis=3)
    pooled_scores_arr = np.stack(pooled_scores_list, axis=3)
    return pooled_scores_arr


def sub_scoring_algs_stack(img_arr_stack, kernel, kernel_pos, algs=None):

    if algs is None:
        algs = ['pooled']

    # calculate shifts
    win_size_left = int((kernel.shape[0] - 1) / 2)
    win_size_right = (kernel.shape[0] - 1) - win_size_left
    total_win_size = win_size_left + win_size_right

    # create padded image
    padded_shape = (img_arr_stack.shape[0] + total_win_size, img_arr_stack.shape[1] + total_win_size, img_arr_stack.shape[2])
    img_arr_padded = np.zeros(shape=padded_shape)
    img_arr_padded[win_size_left:-win_size_right, win_size_left:-win_size_right, :] = img_arr_stack

    # generate "pooled" image
    pooled_image = np.zeros(shape=img_arr_stack.shape)
    mean_pooled_image = np.zeros(shape=img_arr_stack.shape)
    mapped_image = np.zeros(shape=img_arr_stack.shape)
    sub_pooled_image = np.zeros(shape=img_arr_stack.shape)
    sub_ps = 2

    count = 0
    for i in range(0, img_arr_stack.shape[0], 2):
        for j in range(0, img_arr_stack.shape[1]):
            count += 1
            if count % 2 == 0:
                continue

            # print(i, j)
            pi = i + win_size_left
            pj = j + win_size_left

            if 'pooled' in algs:
                patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                        (pj - win_size_left):(pj + win_size_right + 1), :]
                pooled_image[i, j, :] = np.max(patch, axis=(0, 1))

            if 'sub_pooled' in algs:
                pikp = pi - (7 - kernel_pos[0])
                pjkp = pj - (7 - kernel_pos[1])
                if pikp - sub_ps < 0 or pjkp - sub_ps < 0:
                    sub_patch = np.zeros(shape=(1, img_arr_stack.shape[2]))
                else:
                    sub_patch = img_arr_padded[(pikp - sub_ps):(pikp + sub_ps + 1),
                            (pjkp - sub_ps):(pjkp + sub_ps + 1), :]

                sub_pooled_image[i, j, :] = np.max(sub_patch, axis=(0, 1))

            if 'mean_pooled' in algs:
                patch = img_arr_padded[(pi - win_size_left):(pi + win_size_right + 1),
                        (pj - win_size_left):(pj + win_size_right + 1)]
                mean_pooled_image[i, j] = np.mean(patch, axis=(0, 1))

    rt = []
    if 'pooled' in algs:
        rt.append(pooled_image)

    if 'mean_pooled' in algs:
        rt.append(mean_pooled_image)

    if 'sub_pooled' in algs:
        rt.append(sub_pooled_image)

    return rt
