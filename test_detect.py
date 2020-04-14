import numpy as np
import os
from PIL import Image
from run_predictions import detect_red_light
import json
import visualize

print('test_detect')
# set the path to the downloaded data:
data_path = '../data/RedLights2011_test'

# set a path for saving predictions:
preds_path = '../data/hw01_test'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):
    # read image using PIL:
    img = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(img)
    # visualize.visualize_three_channel_image(I)
    preds[file_names[i]] = detect_red_light(I, file_names[i])
    # fig = visualize.visualize_image_with_bounding_boxes(img, preds[file_names[i]])
    # visualize.show()
    # visualize.close(fig)


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)

visualize.visualize_all_images_with_bounding_boxes(data_path, preds_path + '/preds.json')