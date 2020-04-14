from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from archive import matched_filtering as mf


def load_image(img_path):
    img = Image.open(img_path)
    return img


def image_to_array(img):
    return np.asarray(img, dtype=np.float32) / 255


def preprocess(img):
    print()
    # redshift
    im_arr = np.copy(np.asarray(img))
    # im_arr[:, :, 1] = 0
    # im_arr[:, :, 2] = 0

    # threshold
    threshold = 200
    im_arr[im_arr[:, :, 0] < threshold] = 0
    plt.imshow(im_arr)
    plt.show()


def kernel_smoothing(kernel):

    pkernel = np.copy(kernel)

    gaussian_sub_kernel1 = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)
    gaussian_sub_kernel2 = np.array([1/32, 1/16, 1/32, 1/16, 1/4, 1/16, 1/32, 1/16, 1/32]).reshape(3, 3)
    gaussian_sub_kernel3 = generate_gaussian_kernel(s=4, sigma=1)
    gaussian_kernel = np.stack([gaussian_sub_kernel3] * 3, axis=2)
    smoothed_kernel = mf.dot_product_filter(pkernel, gaussian_kernel)
    smoothed_kernel = clip_image(smoothed_kernel, gaussian_kernel)

    return smoothed_kernel


def clip_image(img_arr, kernel):

    win_size = int((kernel.shape[0] - 1) / 2)
    clipped_kernel = img_arr[win_size:, win_size:, :]
    return clipped_kernel


def generate_gaussian_kernel(s=3, sigma=1.0):
    x, y = np.meshgrid(np.linspace(-1, 1, s), np.linspace(-1, 1, s))
    d = np.sqrt(x * x + y * y)
    mu = 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    print(g)
    return g
