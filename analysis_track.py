"""Functions to create a dataframe of the trajectories and calculate their MSD
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_dataframe(local_maxima, liste_a):

    """Create a dataframe and give a name to every tracked particle.

    Parameters
    ----------
    local_maxima : ndarray
        coordinate of local maxima
    liste_a : list
        list of trajectories

    Returns
    -------
    Dataframe : Dataframe

    """

    name =[]
    i=0
    for n in range(len(local_maxima[0])):
        i+=1
        name.append("cell_{}".format(i))

    liste_dataframe =[]
    for x in range(len(liste_a)):
        df = pd.DataFrame(liste_a[x], columns=["x", "y", "t"])
        liste_dataframe.append(df)
    trajectories = pd.concat(liste_dataframe,axis=0, keys=name)

    return(trajectories)

def compute_msd(trajectory, t_step, coords=['x', 'y']):

    """Compute mean square displacement for all possible delays.

    Parameters
    ----------
    trajectory : Dataframe
        Dataframe of trajectories
    t_step : int
        time step between two points
    coords : list
        columns x and y from the df of the trajectories

    Returns
    -------
    delays: ndarray
    MSD: ndarray

    """

    delays = trajectory['t']
    shifts = np.floor(delays/t_step).astype(np.int)
    msds = np.zeros(shifts.size)
    for i, shift in enumerate(shifts):
        diffs = trajectory[coords] - trajectory[coords].shift(-shift)
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = np.trim_zeros(sqdist).mean()
    return delays, msds

def plot_MSD(trajectories, t_step = 1, save=False, name='test.png'):

    """Plot Mean Square Displacement.

    Parameters
    ----------
    trajectory : Dataframe
        Dataframe of trajectories
    t_step : int
        time step between two points
    save : bool
        True if want to save the image
    name : string
        name of the save image

    Return
    -------
    Plot

    """

    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    color=iter(plt.cm.jet(np.linspace(0,1,len(trajectories.groupby(level=0)))))

    for cell, trajectory in trajectories.groupby(level=0):
        c=next(color)
        delays, msds = compute_msd(trajectory, t_step=t_step)
        ax.plot(delays, msds, '-r',label=cell, c=c)
        ax.legend(loc=2, ncol=2, fontsize=18)
        ax.set_title("Random movement", fontsize=18)
        ax.set_xlabel('Delay (s)', fontsize=18)
        ax.set_ylabel('MSD (µm²)', fontsize=18)
    if save == True:
        fig.savefig(name, bbox_inches='tight')
