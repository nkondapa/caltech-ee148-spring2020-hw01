from generate_matched_filter_kernel import *

path = '../data/hw01_preds/preds.json'
kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'

convolved_image = np.load('../data/convolved_images/ci04.npy')
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