import utilities
from visualize import *

save_path = '../data/output/'
utilities.create_nonexistent_folder(save_path)

visualize_all_images_with_bounding_boxes(save_path=save_path)
