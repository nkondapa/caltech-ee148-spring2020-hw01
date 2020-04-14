import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def threshold_convolved_image(img_arr_orig, threshold):

    img_arr = np.copy(img_arr_orig)

    img_arr[img_arr < threshold] = 0
    img_arr[img_arr > threshold] = 1

    return img_arr


def group_pixels(img_arr):

    x, y = np.where(img_arr > 0)

    points = set(zip(x, y))
    points_to_pass = set(zip(x,y))
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
            xl = [x - 1] * 3 + [x] * 3 + [x + 1] * 3
            yl = [y - 1, y, y + 1] * 3
            candidates = set(zip(xl, yl))
            for c in candidates:
                if c in points:
                    points.remove(c)

                    groups[-1].add(c)
                    num_items += 1
                    bfs.append(c)
        if len(groups[-1]) <= 5**2:
            groups.pop(-1)

    return groups, points_to_pass


def groups_to_bounding_boxes(groups):

    bounding_boxes = []
    for group in groups:
        most_left = None
        most_top = None
        most_right = None
        most_bottom = None
        for p in group:
            if most_left is None or p[0] < most_left:
                most_left = p[0]
            if most_top is None or p[1] < most_top:
                most_top = p[1]
            if most_right is None or p[0] > most_right:
                most_right = p[0]
            if most_bottom is None or p[1] > most_bottom:
                most_bottom = p[1]

        bounding_box = [int(most_top), int(most_left), int(most_bottom), int(most_right)]
        bounding_boxes.append(bounding_box)

    return bounding_boxes
