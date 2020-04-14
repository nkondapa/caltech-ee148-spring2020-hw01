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

kernel_id = 0
image_path = image_paths[0]
img = load_image(image_path)
img_arr = image_to_array(img)

kernel = load_kernel(kernel_id, base_kernel_path)

lp1 = save_path + 'kernel=' + str(kernel_id) + '_filter1d.npy'
lp2 = save_path + 'kernel=' + str(kernel_id) + '_filter2d.npy'
lp3 = save_path + 'kernel=' + str(kernel_id) + '_filter3d.npy'

load1d = 0
load2d = 0
load3d = 1

if load1d:
    ci1 = np.load(lp1, allow_pickle=True)

    viz.visualize_convolved_image(ci1)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(np.sum(ci1, axis=2)))
    viz.visualize_three_channel_image(postp.threshold_convolved_image(ci1))

if load2d:
    ci2 = np.load(lp2, allow_pickle=True)
    viz.visualize_convolved_image(ci2)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(np.sum(ci2, axis=2)))
    viz.visualize_three_channel_image(postp.threshold_convolved_image(ci2))


if load3d:
    ci3 = np.load(lp3, allow_pickle=True)
    viz.visualize_one_channel_image(ci3)
    viz.visualize_one_channel_image(postp.threshold_convolved_image(ci3))
