"""Functions to plot the track either in 2D or 3D
"""

import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np

def plot_track(img, liste_a, save=False, name='test.png'):
    """Plot track on the image.

    Parameters
    ----------
    img : ndarray
        The original image.
    liste_a : list
        list of ndarray.
    save : bool
        True if want to save the image
    name : string
        name of the save image

    Return
    -------
    Plot

    """

    nt, ny, nx = img.shape

    fig, axes = plt.subplots(1,1, figsize=(nx/15, ny/15))
    axes.imshow(img[0,:,:], cmap="gray",interpolation='nearest')

    color=iter(plt.cm.jet(np.linspace(0,1,len(liste_a))))

    for n in range(len(liste_a)):
        c=next(color)
        axes.plot(liste_a[n][:,1], liste_a[n][:,0], linewidth=3, c=c)

    axes.axis("off")
    axes.autoscale_view('tight')

    if save == True:
        fig.savefig(name, bbox_inches='tight')

def plot_track_3d(liste_a, save=False, name='test.png'):
    """Plot track on the image.

    Parameters
    ----------
    liste_a : list
        list of ndarray.
    save : bool
        True if want to save the image
    name : string
        name of the save image

    Return
    -------
    Plot

    """

    scene = mlab.figure(size = (1024,768), fgcolor=(0, 0, 0), bgcolor=(0.8, 0.8, 0.8))

    arr = np.concatenate(liste_a[:], axis=0)
    x = arr[:,1]
    y = arr[:,0]
    z = arr[:,2]
    ax = mlab.plot3d(x,y, z, tube_radius=0)
    ax = mlab.axes(zlabel="T")
    ax.title_text_property.bold = False
    ax.label_text_property.bold = False
    ax.title_text_property.italic = False
    ax.label_text_property.italic = False
    ax.title_text_property.font_family = "times"
    ax.label_text_property.font_family = "times"
    ax.axes.font_factor=1
    mlab.outline()

    for n in range(len(liste_a)):
        x = liste_a[n][:,1]
        y = liste_a[n][:,0]
        z = liste_a[n][:,2]
        t = np.linspace(0, 20, len(z))
        exp = mlab.plot3d(x,y, z, t, tube_radius=2)

    scene.scene.camera.position = [723.11244587305441, 595.11034212703476, 474.40094467678716]
    scene.scene.camera.focal_point = [250.2378683140945, 122.2357645680766, 1.5263671178281939]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.scene.camera.clipping_range = [384.60043576038015, 1367.8917952716088]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    #mlab.show()

    arr = mlab.screenshot(mode='rgba')
    mlab.close(scene)

    fig, axes = plt.subplots(figsize=(24,24))
    axes.axis("off")
    axes.autoscale_view('tight')
    axes.imshow(arr)

    if save == True:
        fig.savefig(name, bbox_inches='tight')
