import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def visualize_convolved_image(convolved_image):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(convolved_image)
    plt.subplot(2, 2, 2)
    plt.imshow(convolved_image[:, :, 0])
    plt.subplot(2, 2, 3)
    plt.imshow(convolved_image[:, :, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(convolved_image[:, :, 2])

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(np.product(convolved_image, axis=2))
    plt.subplot(3, 1, 2)
    plt.imshow(np.sum(convolved_image, axis=2))
    plt.show()
    print()


def visualize_three_channel_image(image):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    plt.imshow(image[:, :, 0])
    plt.subplot(2, 2, 3)
    plt.imshow(image[:, :, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(image[:, :, 2])
    plt.show()


def visualize_one_channel_image(image):
    plt.figure()
    plt.imshow(np.squeeze(image))
    plt.show()


def visualize_all_images_with_bounding_boxes(image_base_path=None, prediction_path=None, save_path=None):
    if prediction_path is None:
        prediction_path = '../data/hw01_preds/preds.json'

    if image_base_path is None:
        image_base_path = '../data/RedLights2011_Medium'

    with open(prediction_path, 'r') as f:
        predictions_dict = json.load(f)

    for key in predictions_dict.keys():
        image_path = image_base_path + '/' + key
        img = Image.open(image_path, 'r')
        bounding_boxes = predictions_dict[key]
        print(image_path + ' | ' + str(len(bounding_boxes)))
        if save_path is None:
            visualize_image_with_bounding_boxes(img, bounding_boxes)
            show()
        else:
            save_image_with_bounding_boxes(img, bounding_boxes, save_path + key)


def visualize_image_with_bounding_boxes(img, bounding_boxes):

    draw = ImageDraw.Draw(img)

    for bb in bounding_boxes:
        draw.rectangle(bb)
    return display_image(img)


def save_image_with_bounding_boxes(img, bounding_boxes, save_path):

    draw = ImageDraw.Draw(img)

    for bb in bounding_boxes:
        draw.rectangle(bb)

    img.save(save_path)


def display_image(img, mode='plt'):

    if mode == 'plt':
        fig = plt.figure()
        plt.imshow(np.asarray(img))
        return fig


def show():
    plt.show()


def save(fig, path):
    plt.savefig()


def close(fig=None):
    if fig is not None:
        plt.close(fig)
    else:
        plt.close('all')
