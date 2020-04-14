import numpy as np
import matplotlib.pyplot as plt


def threshold_convolved_image(convolved_image):

    tci = np.copy(convolved_image)

    for i in range(3):
        m = np.mean(convolved_image[:, :, i])
        s = np.std(convolved_image[:, :, i])

        thr = m
        tci[convolved_image[:, :, i] < thr, i] = 0

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(tci[:, :, 0])
    plt.subplot(2, 2, 2)
    plt.imshow(tci[:, :, 1])
    plt.subplot(2, 2, 3)
    plt.imshow(tci[:, :, 2])
    plt.subplot(2, 2, 4)
    plt.imshow(np.sum(tci, axis=2))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(convolved_image[:, :, 0])
    plt.subplot(2, 2, 2)
    plt.imshow(convolved_image[:, :, 1])
    plt.subplot(2, 2, 3)
    plt.imshow(convolved_image[:, :, 2])
    plt.subplot(2, 2, 4)
    plt.imshow(np.sum(convolved_image, axis=2))
    plt.show()
    print()

ci = np.load('../data/test/ci=0.npy')
k = np.load('../data/kernels/kernel=0.npy')
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(k[:, :, 0])
plt.subplot(2, 2, 2)
plt.imshow(k[:, :, 1])
plt.subplot(2, 2, 3)
plt.imshow(k[:, :, 2])

threshold_convolved_image(ci)
