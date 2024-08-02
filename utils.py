import numpy as np
import math
from SegmentationMap import SegmentationMap as SM


def _generate_voronoi_diagram(width, height, centers_x, centers_y, k):
    arr = np.zeros((width, height), dtype=int)
    imgx, imgy = width, height
    num_cells = k

    nx = centers_x
    ny = centers_y

    # randcolors = np.random.randint(0, num_cells, size=(num_cells))
    colors = np.array(range(k))

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i
            arr[x, y] = colors[j]

    return arr


def generate_random_voronoi_prior(width, height, k, end_iter=0.03):
    tol = 1000
    while tol > end_iter:
        cx = np.random.rand(k) * width
        cy = np.random.rand(k) * height
        vd = _generate_voronoi_diagram(width, height, cx, cy, k)
        counts = np.unique(vd, return_counts=True)[1]

        uniformity = np.std(counts / sum(counts))
        tol = uniformity

    return vd


def generate_random_prior(width, height, k):
    out = np.random.choice(range(k), size=(width, height))

    return out


def get_SegMap_pmaps(SegMap, k):
    weights = SegMap._res_iter.squeeze()[:, 1]

    pmaps_per_iter = np.asarray(
        [
            weight.reshape((*SegMap.im.shape[:2], k))
            for weight in weights
            if type(weight) != int
        ]
    )

    pmaps = np.moveaxis(pmaps_per_iter,-1,1)

    return pmaps


def visualize_em(SegMap):
    from matplotlib import pyplot as plt

    weights = SegMap._res_iter.squeeze()[:, 1]

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
    for i, ax in enumerate(np.ravel(axs)):
        ax.imshow(SegMap.im, cmap="gray")
        ax.imshow(weights[i].argmax(-1).reshape((SegMap.im.shape[:2])))
    fig, axs = plt.subplots()
    axs.imshow(SegMap.im)
    axs.imshow(
        weights[-1].argmax(-1).reshape(SegMap.im.shape[:2]), alpha=0.4
    )
    axs.axis("off")
    fig.show()

def get_Response_jaccard(Response):
    k = Response.kSeg
    SegMap  = SM((0,Response.image), mode="array")
    SegMap.fit_model(
        model="c",
        n_components = np.array([k]),
        layer_stop = 1,
        keep = False,
        init = None,
        init_eps
    )