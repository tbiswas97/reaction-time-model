import numpy as np
import math
from SegmentationMap import SegmentationMap as SM


def _generate_voronoi_diagram(width, height, centers_x, centers_y, k):
    """
    Generates a Voronoi diagram based on k polygon centroid coordinates. k must
    match length of centers_x and center_y

    Parameters:
    -----------
    width : int
        the width of the image to generate a prior for
    height : int
        the height of the image to generate a prior for
    centers_x : np.array
        the x-coordinate of the center of the Voronoi polygons
    centers_y : np.array
        the y-coordinate of the center of the Voronoi polygons
    k : int
        The numnber of polygons to generate

    """
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
    """
    Uses RNG to generate the centroids of Voronoi polygons. This process is
    repeated until the number of pixels per polygon is within user-specified
    tolerance

    Parameters:
    ------------
    width : int
        the width of the image to generate a prior for
    height : int
        the height of the image to generate a prior for
    k : int
        the number of polygons in the Voronoi diagram
    end_iter : float
        the tolerance for how close in size the polygons are to each other.
        Note: a value that is too small may fail to converge

    Returns:
    ---------
    vd : np.array
        shape (width, height)

    """
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
    """
    Generates a random prior for the given number of classes. Each pixel is
    assigned an integer between 0 and k by randomly sampling from a uniform
    distribution.

    Parameters:
    ------------
    width : int
        the width of the image to generate a prior for
    height : int
        the height of the image to generate a prior for
    k : int
        the number of components to assign each pixel to

    Returns:
    ---------
    out : np.array
        shape (width, height)
    """
    out = np.random.choice(range(k), size=(width, height))

    return out


def get_SegMap_pmaps(SegMap, k):
    """
    Unravels the pixel weights from the SegmentationMap classes and reshapes
    them into the correct shape.

    Parameters:
    ------------
    SegMap : SegmentationMap object
        This object stores the output of the EM algorithm
    k : int
        the number of components

    Returns:
    ---------
    pmaps : np.ndarray
        shape (n_iter,k,width,height)
    """
    weights = SegMap._res_iter.squeeze()[:, 1]

    pmaps_per_iter = np.asarray(
        [
            weight.reshape((*SegMap.im.shape[:2], k))
            for weight in weights
            if type(weight) != int
        ]
    )

    pmaps = np.moveaxis(pmaps_per_iter, -1, 1)

    return pmaps


# DEPRECATED?
def visualize_em(SegMap):
    from matplotlib import pyplot as plt

    weights = SegMap._res_iter.squeeze()[:, 1]

    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
    for i, ax in enumerate(np.ravel(axs)):
        ax.imshow(SegMap.im, cmap="gray")
        ax.imshow(weights[i].argmax(-1).reshape((SegMap.im.shape[:2])))
    fig, axs = plt.subplots()
    axs.imshow(SegMap.im)
    axs.imshow(weights[-1].argmax(-1).reshape(SegMap.im.shape[:2]), alpha=0.4)
    axs.axis("off")
    fig.show()


def get_Response_jaccard(Response):
    k = Response.kSeg
    SegMap = SM((0, Response.image), mode="array")
    SegMap.fit_model(
        model="c",
        n_components=np.array([k]),
        layer_stop=1,
        keep=False,
        init=None,
    )
