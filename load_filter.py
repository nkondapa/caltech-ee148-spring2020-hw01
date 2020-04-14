import glob
from utilities import *
import visualize as viz
import postprocessing as postp
import matplotlib.pyplot as plt


path = '../data/hw01_preds/preds.json'
base_kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'
save_path = '../data/convolved_images/'

image_paths = sorted(glob.glob(image_base_path + '/*'))

kernel_id = 'average'
image_path = image_paths[0]
img = load_image(image_path)
img_arr = image_to_array(img)

kernel = load_kernel(kernel_id, base_kernel_path)

lp1 = save_path + 'kernel=' + str(kernel_id) + '_filter1d.npy'
lp2 = save_path + 'kernel=' + str(kernel_id) + '_filter2d.npy'
lp3 = save_path + 'kernel=' + str(kernel_id) + '_filter3d.npy'

load1d = 1
load2d = 1
load3d = 1

if load1d:
    ci1 = np.load(lp1, allow_pickle=True)
    ci1 = ci1/np.max(ci1, axis=(0, 1))
    viz.visualize_convolved_image(ci1)
    oc_ci1 = np.sum(ci1, axis=2)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(oc_ci1/np.max(oc_ci1), 0.88))
    viz.visualize_three_channel_image(postp.threshold_convolved_image(ci1, 0.88))

if load2d:
    ci2 = np.load(lp2, allow_pickle=True)
    ci2 = ci2/np.max(ci2, axis=(0, 1))
    viz.visualize_convolved_image(ci2)

    oc_ci2 = np.sum(ci2, axis=2)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(oc_ci2/np.max(oc_ci2), 0.88))
    viz.visualize_three_channel_image(postp.threshold_convolved_image(ci2, 0.88))


if load3d:
    ci3 = np.load(lp3, allow_pickle=True)
    ci3 = ci3 / np.max(ci3)
    viz.visualize_one_channel_image(ci3)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(ci3, 0.88))
