from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path):
    img = Image.open(img_path)
    return img


def image_to_array(img):
    return np.asarray(img)


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
