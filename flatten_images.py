"""Functions for illumination correction of a serie of images and plot the
    result.

S. Singh; M.-A. Bray; T.R. Jones; A.E. Carpenter
Pipeline for illumination correction of images for high-throughput microscopy.
Journal of microscopy. 2014,

"""

import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
import numpy as np
from scipy.interpolate import spline

def illumin_correct(image, radius=20):
    """Correct for uneven illumination.

    Parameters
    ----------
    image : ndarray
        The input image.
    radius : int
        The radius of the disk-shaped structuring element.

    Returns
    -------
    Divided image : ndarray
        Flattened image

    """
    nt, nx, ny = image.shape
    img_average = np.mean(image, axis = 0).astype('uint8')
    img_median = median(img_average, disk(radius))
    img_divide = np.empty([nt,nx,ny])
    for t in range(nt):
        img_divide[t,:,:] = np.divide(image[t,:,:], img_median)
    return(img_divide)

def plot_profile_corrected_image(img, img_divide, position = 50, div=1,
                                 save=False, name='test.png'):
    """Plot a line profile of the original image and the flattened image.

    Parameters
    ----------
    img : ndarray
        The original image.
    img_divide : ndarray
        The corrected image.
    position : int
        Position in y of the line profile.
    div : int
        Can smooth the line profile
    save : bool
        True if want to save the image
    name : string
        name of the save image

    Returns
    -------
    Plot

    """

    fig, axes = plt.subplots(2,2, figsize=(24, 12))

    x_smooth_divide, y_smooth_divide=smooth_plot(img_divide,
                                                 position = position, div = div)
    x_smooth, y_smooth=smooth_plot(img, position = position, div = div)

    axes[0,0].imshow(img[0,:,:], cmap="gray",interpolation='nearest')
    axes[0,0].plot([0,len(img[0,position,:-1])],
                   [position, position], 'k-', color='r')
    axes[0,0].axis("off")
    axes[0,0].autoscale_view('tight')
    axes[1,0].plot(x_smooth, y_smooth)

    axes[0,1].imshow(img_divide[0,:,:], cmap="gray",interpolation='nearest')
    axes[0,1].plot([0,len(img_divide[0,position,:-1])],
                   [position, position], 'k-', color='r')

    axes[0,1].axis("off")
    axes[0,1].autoscale_view('tight')
    axes[1,1].plot(x_smooth_divide, y_smooth_divide)

    if save == True:
        fig.savefig(name, bbox_inches='tight')

def smooth_plot(image, position = 50, div=1):

    """Plot a line profile of the original image and the flattened image.

    Parameters
    ----------
    img : ndarray
        The original image.
    position : int
        Position in y of the line profile.
    div : int
        Can smooth the line profile

    Returns
    -------
    x_smooth, y_smooth : ndarray

    """

    x_no_smooth = np.linspace(0, len(image[0,position,:-1]),
                              len(image[0,position,:-1]))
    x_smooth = np.linspace(0, len(image[0,position,:-2]),
                           len(image[0,position,:-1])/div)
    y_smooth = spline(x_no_smooth, image[0,position,:-1], x_smooth)

    return(x_smooth, y_smooth)
