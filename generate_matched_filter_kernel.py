import numpy as np
import matplotlib.pyplot as plt
import visualize as viz
import utilities
import glob


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
    box_size = 6
    boxes = []
    for i, event in enumerate(events):
        # WHY IS THIS SHIT ALWAYS INVERTED
        x = int(event.xdata)
        y = int(event.ydata)
        print(x, y)
        print(img.getpixel(xy=(x, y)))
        box = im_arr[(y-box_size):(y+box_size+1), (x-box_size):(x+box_size+1), :]
        boxes.append(box)
        plt.imshow(box)
        plt.show()
        save_kernels(box, kernel_save_path, i)

    np.stack(boxes, axis=3)
    box_stack = np.stack(boxes, axis=3)
    avg_box = np.mean(box_stack, axis=3)
    save_kernels(avg_box, kernel_save_path, 'average')

    fig.canvas.mpl_disconnect(cid)


def inspect_image(img):

    events = []
    img_arr = np.asarray(img)

    def onclick(event):
        ex = int(event.xdata)
        ey = int(event.ydata)
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, rgb=(%d, %d, %d)' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata, img_arr[ey, ex, 0], img_arr[ey, ex, 1], img_arr[ey, ex, 2]))

        events.append(event)

    fig = plt.figure()
    plt.imshow(img_arr)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    viz.plt.show()

    im_arr = np.copy(np.asarray(img))
    print(im_arr.shape)
    # im_mask = np.zeros(im_arr.shape, dtype=np.bool)
    box_size = 6
    for i, event in enumerate(events):
        # WHY IS THIS SHIT ALWAYS INVERTED
        x = int(event.xdata)
        y = int(event.ydata)
        print(x, y)
        print(img.getpixel(xy=(x, y)))
        box = im_arr[(y-box_size):(y+box_size+1), (x-box_size):(x+box_size+1), :]
        plt.imshow(box)
        plt.show()
    fig.canvas.mpl_disconnect(cid)


def save_kernels(box, kernel_save_path, i):
    utilities.create_nonexistent_folder(kernel_save_path)
    np.save(kernel_save_path + '/kernel=' + str(i), box)


path = '../data/hw01_preds/preds.json'
kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'


image_paths = sorted(glob.glob(image_base_path + '/*'))

image_path = image_paths[0]
img = utilities.load_image(image_path)
img_arr = utilities.image_to_array(img)
search_for_kernel(img, kernel_path)
