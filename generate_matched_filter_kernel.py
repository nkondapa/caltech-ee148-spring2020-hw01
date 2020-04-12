import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import visualize as viz
import utilities


def search_for_kernel(img, kernel_save_path):

    events = []

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

        events.append(event)

    fig = plt.figure()
    plt.imshow(np.asarray(img))
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    viz.plt.show()

    im_arr = np.copy(np.asarray(img))
    print(im_arr.shape)
    # im_mask = np.zeros(im_arr.shape, dtype=np.bool)
    box_size = 5
    for i, event in enumerate(events):
        # WHY IS THIS SHIT ALWAYS INVERTED
        x = int(event.xdata)
        y = int(event.ydata)
        print(x, y)
        print(img.getpixel(xy=(x, y)))
        box = im_arr[(y-box_size):(y+box_size), (x-box_size):(x+box_size), :]
        plt.imshow(box)
        plt.show()
        save_kernels(box, kernel_save_path, i)
    fig.canvas.mpl_disconnect(cid)


def save_kernels(box, kernel_save_path, i):
    utilities.create_nonexistent_folder(kernel_save_path)
    np.save(kernel_save_path + '/kernel=' + str(i), box)




