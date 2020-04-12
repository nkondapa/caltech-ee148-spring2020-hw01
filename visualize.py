import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def visualize_all_images_with_bounding_boxes():
    path = '../data/hw01_preds/preds.json'
    image_base_path = '../data/RedLights2011_Medium'

    with open(path, 'r') as f:
        predictions_dict = json.load(f)

    for key in predictions_dict.keys():
        image_path = image_base_path + '/' + key
        img = Image.open(image_path, 'r')
        draw = ImageDraw.Draw(img)
        bounding_boxes = predictions_dict[key]
        print(image_path + ' | ' + str(len(bounding_boxes)))
        for bb in bounding_boxes:
            draw.rectangle(bb)
        display_image(img)


def display_image(img, mode='plt'):

    if mode == 'plt':
        fig = plt.figure()
        plt.imshow(np.asarray(img))
        return fig


def show():
    plt.show()


def close(fig=None):
    if fig is not None:
        plt.close(fig)
    else:
        plt.close('all')
