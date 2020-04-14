import glob
from utilities import *
import pixel_distance as pd


path = '../data/hw01_preds/preds.json'
base_kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'
save_path = '../data/convolved_images/'

image_paths = sorted(glob.glob(image_base_path + '/*'))

image_path = image_paths[0]
img = load_image(image_path)
img_arr = image_to_array(img)

kernel_id = 0
kernel1 = load_kernel(0, base_kernel_path)
kernel2 = load_kernel(3, base_kernel_path)
kernel_stack = np.stack([kernel1, kernel2], axis=3)
pd.pixel_rgb_distance_map_multi_kernel(img_arr, kernel_stack)