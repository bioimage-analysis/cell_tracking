"""Functions for detecting and tracking blobs in images, can work with time lapse.

"""



import math
import numpy as np
from scipy.ndimage import gaussian_filter
import itertools as itt
from scipy.spatial import distance


def find_max(image, x):
    """Find maximum intensity.

    Parameters
    ----------
    image : ndarray
        The input image.
    x : int

    Returns
    -------
    max intensity : ndarray

    """
    J = np.zeros(image.shape, dtype=bool)
    J[1:-1,1:-1,x] = ((image[1:-1,1:-1,x] > image[ :-2, :-2,x]) & (image[1:-1,1:-1,x] > image[ :-2, 1:-1,x]) & (image[1:-1,1:-1,x] > image[ :-2,2:,x]) &
                      (image[1:-1,1:-1,x] > image[1:-1, :-2,x]) &                                      (image[1:-1,1:-1,x] > image[1:-1,2:,x]) &
                      (image[1:-1,1:-1,x] > image[2:  , :-2,x]) & (image[1:-1,1:-1,x] > image[2:  , 1:-1,x]) & (image[1:-1,1:-1,x] > image[2:  ,2:,x]))

    J[1:-1,1:-1,(x-1)] = ((image[1:-1,1:-1,x] > image[ :-2, :-2,(x-1)]) & (image[1:-1,1:-1,x] > image[ :-2, 1:-1,(x-1)]) & (image[1:-1,1:-1,x] > image[ :-2,2:,(x-1)]) &
                      (image[1:-1,1:-1,x] > image[1:-1, :-2,(x-1)]) & (image[1:-1,1:-1,x] > image[1:-1, 1:-1,(x-1)]) & (image[1:-1,1:-1,x] > image[1:-1,2:,(x-1)]) &
                      (image[1:-1,1:-1,x] > image[2:  , :-2,(x-1)]) & (image[1:-1,1:-1,x] > image[2:  , 1:-1,(x-1)]) & (image[1:-1,1:-1,x] > image[2:  ,2:,(x-1)]))

    J[1:-1,1:-1,(x+1)] = ((image[1:-1,1:-1,x] > image[ :-2, :-2,(x+1)]) & (image[1:-1,1:-1,x] > image[ :-2, 1:-1,(x+1)]) & (image[1:-1,1:-1,x] > image[ :-2,2:,(x+1)]) &
                      (image[1:-1,1:-1,x] > image[1:-1, :-2,(x+1)]) & (image[1:-1,1:-1,x] > image[1:-1, 1:-1,(x+1)]) & (image[1:-1,1:-1,x] > image[1:-1,2:,(x+1)]) &
                      (image[1:-1,1:-1,x] > image[2:  , :-2,(x+1)]) & (image[1:-1,1:-1,x] > image[2:  , 1:-1,(x+1)]) & (image[1:-1,1:-1,x] > image[2:  ,2:,(x+1)]))

    return J[:,:,x] & J[:,:,x-1] & J[:,:,x+1]



def find_blob_DoG(image, thresh, dist, min_sigma=1, max_sigma=100, sigma_ratio=1.6):

    """Find blob in the image (Most of the fuction come from blob_dog in scikit-image).

    Blobs are found using the Difference of Gaussian (DoG) method.
    For each blob found, the method returns its coordinates.

    Parameters
    ----------
    image : ndarray.
        Input grayscale image.
    min_sigma : float, optional.
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional.
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    sigma_ratio : float, optional.
        The ratio between the standard deviation of Gaussian Kernels used for
        computing the Difference of Gaussians
    thresh: float.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    dist: int.
        Min distance between two blobs, use to make sure we don't detect more
        than one blob per object.

    Returns
    -------
    list of blobs : ndarray

    """

    min_sigma=min_sigma
    max_sigma=max_sigma
    sigma_ratio=sigma_ratio

    k = int(math.log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

        # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)for i in range(k + 1)])

    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
    dog_images = [(gaussian_images[i] - gaussian_images[i + 1])* sigma_list[i] for i in range(k)]
    image_cube = np.dstack(dog_images)

    maxima = []
    for n in range(1, 4):
        maxima.append(find_max(image_cube, n))
    arr = np.array(maxima)


    arr &= image > thresh

    local_maxima = np.argwhere(np.bitwise_or.reduce(arr))
    add = np.zeros(local_maxima[:,0].shape)
    local_maxima = np.concatenate((local_maxima, add[:,np.newaxis]), axis=1).astype("int64")


    for blob1, blob2 in itt.combinations(local_maxima, 2):
        d = math.hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])
        if d < dist :
            blob2[2] = -1

    return np.array([b for b in local_maxima if b[2] >= 0])

def Num_Blob(data, nt, thresh=1.5, dist=5):

    """To be use if working with 3D data.

    Parameters
    ----------
    data : ndarray.
        Input grayscale image.
    nt : int.
        value of the 3rd dimension

    thresh: float.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    dist: int.
        Min distance between two blobs, use to make sure we don't detect more
        than one blob per object.

    Returns
    -------
    list of blobs : ndarray
    """

    list_blob = []
    for t in range(nt):
        blob = find_blob_DoG(data[t,:,:], thresh=thresh, dist=dist)
        list_blob.append(blob)
    return list_blob

def Construct_Track(local_maxima, max_search = 4, max_dist = 7, particle = 0,
                    nx = 0, tp=0):

    """Function use to track the blobs. The blobs are tracked bases in their
    distance between frame.

    Parameters
    ----------
    local_maxima : ndarray
        list of coordinates of blobs.
    max_search : int
        How many frame we search for a blob within max_dist.
    max_dist : int
        maximum distance to search for a blob.
    particle : int
        Which blob to track in the list of local maxima.
    nx : int
        Which sequence of the local maxima list are we working with.
    tp : Which time point are starting from.

    Returns
    -------
    Tracked particle : list
        list of coordinates.
    """


    lst_fina=[]
    j = 0
    tp=tp
    # nx is the object to track

    l1 = local_maxima[nx][particle,0:2][np.newaxis, :]

    for l2 in local_maxima[1:]:

        # Find all the distances between l1 and every object in the +1 frame
        dist = distance.cdist(l1, l2[:,0:2], 'euclidean')
        #Concatenate coordinate Time frame 1 and +1 if distance is < max
        result =  np.concatenate((l2[:,0][dist[0]<max_dist], l2[:,1][dist[0]<max_dist]), axis=0)
        result_tp = np.concatenate((result, (np.full(1 ,tp, dtype=result.dtype))))

        #check, if during this loop I didn't find any close coordinate at T+1 for max_search above T
        if int(result.shape[0]) == 0:
            j+=1
            if j < max_search:
                continue
            else:
                break
        #if shape >2 (if I found more than 1 objecte within the max distance I stop the loop)
        elif int(result.shape[0]) > 2:
            break

        l1_tp = np.concatenate((l1, (np.full((1,1) ,tp, dtype=l1.dtype))), axis=1)
        lst_fina.append(np.vstack((l1_tp, result_tp)))

        #So I start the loop at T+1

        l1 = result[np.newaxis, :]

        tp+=1
    arr = np.asarray(lst_fina)
    return(arr[:,0,:])

def multiple_track_all(local_maxima, img, max_search = 4, max_dist = 7):

    """Function use to track the blobs. The blobs are tracked bases in their
    distance between frame.

    Parameters
    ----------
    local_maxima : ndarray
        list of coordinates of blobs.
    img : ndarray
        gray scale image used to acquire the local maxima.
    max_search : int
        How many frame we search for a blob within max_dist.
    max_dist : int
        maximum distance to search for a blob.

    Returns
    -------
    List of tracked particle : list
        list of coordinates.
    """



    liste_a=[]
    nt, nx, ny = img.shape

    # First I track every object present in the first frame and create a list
    for x in range(len(local_maxima[0])):
        try:
            Track = Construct_Track(local_maxima, max_search=max_search, max_dist=max_dist,particle = x)
        except IndexError:
            pass

        liste_a.append(Track)
    k=0
    # It will now continue to the next frame

    #First loop is too go through every time point

    for y in range(len(local_maxima)):
        k+=1
        #make sure I stop the loop before I am above the size of local_maxima
        if k >= len(local_maxima):
            break

        #2nd loop is to go through every object in the time frame
        for x in range(len(local_maxima[k])):
            liste_a_tup = [tuple(row) for row in np.vstack(liste_a)[:,0:2]]
            #Check if object is not in the list of coordinate already tracked to avoid duplicate

            if tuple(local_maxima[k][x,0:2]) not in liste_a_tup:
                try:
                    Track2 = Construct_Track(local_maxima, max_search=max_search, max_dist=max_dist,
                                             particle = x, nx = k, tp=y)

                    if np.all(Track2[:,2] <= nt) and len(Track2) > 5:
                        liste_a.append(Track2)
                except IndexError:
                    pass
    return(liste_a)
