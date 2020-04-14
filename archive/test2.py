import glob

from archive.preprocessing import *
from generate_matched_filter_kernel import *
from archive import matched_filtering as mf

path = '../data/hw01_preds/preds.json'
kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'


image_paths = sorted(glob.glob(image_base_path + '/*'))

image_path = image_paths[0]
img = load_image(image_path)
img_arr = image_to_array(img)
# search_for_kernel(img, kernel_path)
kernel = mf.load_kernel(0, kernel_path)
# skernel = kernel_smoothing(kernel)
convolved_image = mf.dot_product_filter(img_arr, kernel)
np.save('../data/convolved_images/ci04', convolved_image)
viz.visualize_convolved_image(convolved_image)
# for i in range(3):
#     fk = np.flip(kernel[:, :, i]).transpose()
#     k2 = kernel[:, :, i]
#
#     b = fk * k2
#     a = (fk/np.sum(fk) + k2/np.sum(k2))
#     plt.figure()
#
#     plt.subplot(3, 2, 1)
#     plt.imshow(fk)
#
#     plt.subplot(3, 2, 2)
#     plt.imshow(k2)
#
#     plt.subplot(3, 2, 3)
#     plt.imshow(a)
#
#     plt.subplot(3, 2, 4)
#     plt.imshow(b)
#
#     plt.subplot(3, 2, 5)
#     plt.imshow(fk/np.sum(fk))
#
# plt.show()

print()