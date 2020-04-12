import glob

from preprocessing import *
from generate_matched_filter_kernel import *
import matched_filtering as mf

path = '../data/hw01_preds/preds.json'
kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'


image_paths = sorted(glob.glob(image_base_path + '/*'))

image_path = image_paths[0]
img = load_image(image_path)
img_arr = image_to_array(img)
kernel = mf.load_kernel(0, kernel_path)
mf.filter(img_arr, kernel)
# preprocess(img)
# search_for_kernel(img, kernel_path)
