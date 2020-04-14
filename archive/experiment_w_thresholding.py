import glob
from utilities import *
from collections import deque
import matplotlib.pyplot as plt
import matched_filter as mf
import archive.preprocessing as app
import visualize as viz

path = '../data/hw01_preds/preds.json'
base_kernel_path = '../data/kernels'
image_base_path = '../data/RedLights2011_Medium'
save_path = '../data/convolved_images/'
save_path2 = '../data/pooled_images/'

kernel_id = 'average'
image_paths = sorted(glob.glob(image_base_path + '/*'))
image_path = image_paths[0]
image = load_image(image_path)
img_arr = image_to_array(image)

lp1 = save_path + 'kernel=' + str(kernel_id) + '_filter1d.npy'
lp2 = save_path + 'kernel=' + str(kernel_id) + '_filter2d.npy'
lp3 = save_path + 'kernel=' + str(kernel_id) + '_filter3d.npy'

ci1 = np.load(lp1, allow_pickle=True)
ci2 = np.load(lp2, allow_pickle=True)
ci3 = np.load(lp3, allow_pickle=True)

pooled_arr = np.load(save_path2 + 'pooled_scores.npy')
mapped_arr = np.load(save_path2 + 'mapped_scores.npy')
sub_pooled_arr = np.load(save_path2 + 'sub_pooled_scores.npy')
multi_kernel_sub_pooled_arr = np.load(save_path2 + 'multi_kernel_sub_pooled_scores.npy')

pooled = np.mean(pooled_arr, axis=2)
mapped = np.mean(mapped_arr, axis=2)
sub_pooled = np.mean(sub_pooled_arr, axis=2)

s = np.mean(multi_kernel_sub_pooled_arr, axis=(3))  # * np.mean(ci1, axis=2)
# threshs = np.linspace(np.mean(s), 1, 21)
# for thr in threshs:
#     pci = np.copy(s)
#     print(thr)
#     pci[pci < thr] = 0
#     pci[pci > thr] = 1
#     # plt.figure()
#     # plt.imshow(np.expand_dims(pci[:, :, 0], axis=2) * img_arr)
#     plt.figure()
#     plt.imshow(np.expand_dims(pci[:, :, 1], axis=2) * img_arr)
#     plt.show()


a = np.product(s, axis=2)
utimg = a / np.max(a)
# timg = mf.smooth(utimg, kernel)
# mtimg = timg/np.max(timg)
utimg[utimg < 0.88] = 0
utimg[utimg > 0.88] = 1
kernel = app.generate_gaussian_kernel()
# mtimg = mf.smooth(utimg, kernel)
mtimg = np.load('../temp.npy')
# np.save('../temp.npy', mtimg)
mtimg = mtimg/np.max(mtimg)
mtimg[mtimg > 0] = 1
x, y = np.where(mtimg == 1)

plt.imshow(mtimg)
plt.show()

points = set(zip(x, y))
bfs = deque()
groups = []
num_items = 0
while len(points) > 0:
    start_point = points.pop()
    bfs.append(start_point)
    groups.append(set())
    groups[-1].add(start_point)
    num_items += 1
    while len(bfs) > 0:
        x, y = bfs.popleft()
        xl = [x-1] * 3 + [x] * 3 + [x+1] * 3
        yl = [y-1, y, y + 1] * 3
        candidates = set(zip(xl, yl))
        for c in candidates:
            if c in points:
                points.remove(c)

                groups[-1].add(c)
                num_items += 1
                bfs.append(c)

plt.imshow(np.expand_dims(mtimg, 2) * img_arr)

tl = (0, 0)
br = img_arr.shape[:2]
bounding_boxes = []
for group in groups:
    best_tl_dist = None
    best_br_dist = None
    best_tl_ind = None
    best_br_ind = None
   #  print(group)
    most_left = None
    most_top = None
    most_right = None
    most_bottom = None
    for p in group:
        # print(p)
        if most_left is None or p[0] < most_left:
            most_left = p[0]
        if most_top is None or p[1] < most_top:
            most_top = p[1]
        if most_right is None or p[0] > most_right:
            most_right = p[0]
        if most_bottom is None or p[1] > most_bottom:
            most_bottom = p[1]
        # tl_dist = (tl[0] - p[0]) ** 2 + (tl[1] - p[1]) ** 2
        # br_dist = (br[0] - p[0]) ** 2 + (br[1] - p[1]) ** 2
        #
        # if best_tl_dist is None or best_tl_dist > tl_dist:
        #     best_tl_dist = tl_dist
        #     best_tl_ind = p
        #
        # if best_br_dist is None or best_br_dist > br_dist:
        #     best_br_dist = br_dist
        #     best_br_ind = p

    # bounding_box = [best_tl_ind[1], best_tl_ind[0], best_br_ind[1], best_br_ind[0]]
    # bounding_boxes.append(bounding_box)

    bounding_box = [most_top, most_left, most_bottom, most_right]
    bounding_boxes.append(bounding_box)

viz.visualize_image_with_bounding_boxes(image, bounding_boxes)
viz.show()