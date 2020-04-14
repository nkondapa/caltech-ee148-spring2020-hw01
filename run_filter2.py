import glob
from utilities import *
import matched_filter as mf
import pixel_distance as pd


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
img_arr = pd.pixel_rgb_distance_map(img_arr)

run1d = 0
run2d = 0
run3d = 1

if run1d:
    ci1 = mf.filter_1d(img_arr, kernel)
    np.save(save_path + 'kernel=' + str(kernel_id) + '_filter1d', ci1)

if run2d:
    ci2 = mf.filter_2d(img_arr, kernel)
    np.save(save_path + 'kernel=' + str(kernel_id) + '_filter2d', ci2)

if run3d:
    ci3 = mf.filter_3d(img_arr, kernel)
    np.save(save_path + 'kernel=' + str(kernel_id) + '_filter3d', ci3)
