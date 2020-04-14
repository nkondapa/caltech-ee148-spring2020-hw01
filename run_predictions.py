import os
import numpy as np
import json
from PIL import Image
import time
import utilities as u
import pixel_distance_algorithms as pda
import matched_filter as mf
import postprocessing as postp


def detect_red_light(I, name=''):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    print(name)
    base_kernel_path = '../data/kernels'
    pooled_scores_path = '../data/pooled_scores/'
    smoothed_image_path = '../data/smoothed_images/'
    u.create_nonexistent_folder(pooled_scores_path)
    u.create_nonexistent_folder(smoothed_image_path)
    # save_path = '../data/'
    kernel_ids = [6, 'average']
    kernels = []
    for id in kernel_ids:
        kernels.append(u.load_kernel(id, base_kernel_path))

    kernel_stack = np.stack(kernels, axis=3)
    img_arr = I / 255

    force_pool = 1
    force_smooth = 1

    st = time.time()

    psp = pooled_scores_path + name.rstrip('.jpg') + '_pooled'
    if not force_pool and u.check_if_file_exists(psp + '.npy'):
        print('Loading pixel matching...')
        sub_pooled_scores_stack = u.numpy_load(psp + '.npy')
    else:
        print('Running pixel matching...')
        sub_pooled_scores_stack = pda.pixel_rgb_distance_map_multi_kernel(img_arr, kernel_stack)
        u.numpy_save(psp, sub_pooled_scores_stack)

    average_pool_per_kernel = np.mean(sub_pooled_scores_stack, axis=3)
    prod_img = np.product(average_pool_per_kernel, axis=2)
    nimg = prod_img / np.max(prod_img)
    timg = postp.threshold_convolved_image(nimg, 0.93)
    gauss_kernel = u.generate_gaussian_kernel(s=5)

    sip = smoothed_image_path + name.rstrip('.jpg') + '_smoothed'
    if not force_smooth and u.check_if_file_exists(sip + '.npy'):
        print('Loading smoothing...')
        simg = u.numpy_load(sip + '.npy')
    else:
        print('Running smoothing...')
        simg = mf.smooth(timg, gauss_kernel)
        u.numpy_save(sip, simg)

    simg = simg / np.max(simg)
    fimg = postp.threshold_convolved_image(simg, np.mean(simg) + np.std(simg))
    groups, points = postp.group_pixels(fimg)
    bounding_boxes = postp.groups_to_bounding_boxes(groups)
    print(time.time() - st)

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes
#
# # set the path to the downloaded data:
# data_path = '../data/RedLights2011_Medium'
#
# # set a path for saving predictions:
# preds_path = '../data/hw01_preds'
# os.makedirs(preds_path,exist_ok=True) # create directory if needed
#
# # get sorted list of files:
# file_names = sorted(os.listdir(data_path))
#
# # remove any non-JPEG files:
# file_names = [f for f in file_names if '.jpg' in f]
#
# preds = {}
# for i in range(len(file_names)):
#
#     # read image using PIL:
#     I = Image.open(os.path.join(data_path,file_names[i]))
#
#     # convert to numpy array:
#     I = np.asarray(I)
#
#     preds[file_names[i]] = detect_red_light(I, file_names[i])
#
# # save preds (overwrites any previous predictions!)
# with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
#     json.dump(preds, f)
