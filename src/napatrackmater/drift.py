
import numpy as np
import dask
import dask.array as da
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from skimage.transform import AffineTransform
from scipy.ndimage import affine_transform, rotate 
# Utility
from tqdm import tqdm
import tifffile 
import sys
import os

def get_xy_drift(_data, ref_channel):
    xy_movie = da.average(_data[ref_channel], axis = 1).compute()

    # correct XY, relative to first frame:
    print('Determining drift in XY')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie)-1)):        

        s, e, p = phase_cross_correlation(xy_movie[t],
                                            xy_movie[t+1], 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
        
    shifts_xy = np.cumsum(np.array(shifts), axis = 0)
    shifts_xy = shifts_xy.tolist()
    shifts_xy.insert(0,[0,0])
    shifts_xy = np.asarray(shifts_xy)
    
    plt.title('XY-Drift')
    plt.plot(shifts_xy[:,0], label = 'x')
    plt.plot(shifts_xy[:,1], label = 'y')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('XY-Drift.svg')
    plt.clf()
    return shifts_xy

def get_z_drift(_data, ref_channel):
    z_movie = da.average(da.swapaxes(_data[ref_channel], 2,1), axis = 1).compute() 

    # correct XY, relative to first frame:
    print('Determining drift in Z')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(z_movie)-1)):        

        s, e, p = phase_cross_correlation(  z_movie[t],
                                            z_movie[t+1], 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
        
    shifts_z = np.cumsum(np.array(shifts), axis = 0)
    for i in shifts_z:
        i[1] = 0

    shifts_z = shifts_z.tolist()
    shifts_z.insert(0,[0,0])
    shifts_z = np.asarray(shifts_z)


    plt.title('Z-Drift')
    plt.plot(shifts_z[:,0], label = 'z')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('Z-Drift.svg')
    plt.clf()

    return shifts_z

def translate_stack(_image, _shift): 

    # define warp matrix:
    tform = AffineTransform(translation=-_shift)

    # construct shifted stack
    _image_out = []
    
    for c,C in enumerate(_image):
        _out_stack = []
        for z,Z in enumerate(C):
            
            # here is the delayed function that allows to scatter everthing nicely across workers
            _out_stack.append(dask.delayed(affine_transform)(Z, np.array(tform)))
        
        # format arrays for proper export
        _out_stack = da.stack([da.from_delayed(x, shape=np.shape(Z), dtype=np.dtype(Z)) for x in _out_stack])
        _image_out.append(_out_stack)

    return da.stack(_image_out)


def apply_xy_drift(_data, xy_drift):

    # Expectes _data to have the following shape:
    # np.shape_data = (channel, time, z, y, x)

    # Swap axes to iterate over time which aligns with chunks
    _data = da.swapaxes(_data, 0,1)
    
    print('Scheduling tasks...')
    _translated_data = []
    for t,T in tqdm(enumerate(_data)):
        _translated_data.append(translate_stack(T, xy_drift[t]))

    # data formatting
    _data_out = da.stack(_translated_data)
    # just in case
    _data = da.swapaxes(_data, 0,1)
    _data_out = da.swapaxes(_data_out, 0,1)

    print('XY-shift has been scheduled.')
    
    return _data_out

def rotate_stack(_image, _alpha):

    # construct shifted stack
    _image_out = []
    
    for c,C in enumerate(_image):
        _out_stack = []
        for z,Z in enumerate(C):
            
            # here is the delayed function that allows to scatter everthing nicely across workers
            _out_stack.append(dask.delayed(rotate)(Z, -_alpha,reshape=False))
        
        # format arrays for proper export
        _out_stack = da.stack([da.from_delayed(x, shape=np.shape(Z), dtype=np.dtype(Z)) for x in _out_stack])
        _image_out.append(_out_stack)

    return da.stack(_image_out)
    


def apply_z_drift(_data, z_drift):
    
    # swap axes around
    _data = da.swapaxes(_data, 0,1)
    _data = da.swapaxes(_data, 3,2)

    _data_out = []
    for t,T in tqdm(enumerate(_data)):
        _data_out.append(translate_stack(T, z_drift[t]))

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data,2,3)
    _data = da.swapaxes(_data,0,1)
    _data_out = da.swapaxes(_data_out,2,3)
    _data_out = da.swapaxes(_data_out,0,1)

    return _data_out

def crop_data(data, xy_drift, z_drift):
    y_crop = [int(np.max(xy_drift[:,0])),int(np.shape(data)[-1]-abs(np.min(xy_drift[:,0])))]
    x_crop = [int(np.max(xy_drift[:,1])),int(np.shape(data)[-2]-abs(np.min(xy_drift[:,1])))]
    z_crop = [int(np.max(z_drift[:,0])),int(np.shape(data)[-3]-abs(np.min(z_drift[:,0])))]

    cropped = data[:,:,
                                                z_crop[0]:z_crop[1],
                                                y_crop[0]:y_crop[1],
                                                x_crop[0]:x_crop[1]]
    return cropped

def get_rotation(data, ref_channel):
    xy_movie = da.average(data[ref_channel], axis = 1).compute()
    
    # get image radius
    radius = int(min(da.shape(data)[3], da.shape(data)[4])/2)

    print('Determining rotation')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie)-1)):        

        s, e, p = phase_cross_correlation(  warp_polar(xy_movie[t], radius = radius),
                                            warp_polar(xy_movie[t+1], radius = radius), 
                                            normalization=None) 
        
        

        shifts.append(s)
        error.append(e)
        phasediff.append(p)
   
    shifts_a = np.cumsum(np.array(shifts), axis = 0)
    for i in shifts: 
        i[0] = 0
    
    shifts_a = shifts_a.tolist()
    shifts_a.insert(0,[0,0])
    shifts_a = np.asarray(shifts_a)
    
    plt.title('alpha-Drift')
    plt.plot(shifts_a[:,0], label = 'alpha')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Rotation in degree')
    plt.savefig('Rotation-Drift.svg')
    plt.clf()

    return shifts_a[:,0]

def apply_alpha_drift(_data, alpha_drift):

    # swap axes around
    _data = da.swapaxes(_data, 0,1)

    _data_out = []
    for t,T in tqdm(enumerate(_data)):
        _data_out.append(rotate_stack(T, alpha_drift[t]))

    # Stack the list to a dask array and swap back the axes.
    _data_out = da.stack(_data_out)
    _data = da.swapaxes(_data,0,1)
    _data_out = da.swapaxes(_data_out,0,1)


    return _data_out

def get_xy_drift_numpy(_data, ref_channel=0):
    """
    Compute XY drift for NumPy array shape (channel, time, z, y, x)
    Returns numpy array of shape (time, 2)
    """
    # replicate da.average
    xy_movie = np.average(_data[ref_channel], axis=1)

    print('Determining drift in XY')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie) - 1)):
        s, e, p = phase_cross_correlation(
            xy_movie[t], xy_movie[t + 1], normalization=None
        )
        shifts.append(s)
        error.append(e)
        phasediff.append(p)

    # cumulative sum
    shifts_xy = np.cumsum(np.array(shifts), axis=0)
    # set up list form
    shifts_xy = shifts_xy.tolist()
    shifts_xy.insert(0, [0, 0])
    shifts_xy = np.asarray(shifts_xy)

    # replicate plotting
    plt.title('XY-Drift')
    plt.plot(shifts_xy[:, 0], label='x')
    plt.plot(shifts_xy[:, 1], label='y')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('XY-Drift.svg')
    plt.clf()

    return shifts_xy


def get_z_drift_numpy(_data, ref_channel=0):
    """
    Compute Z drift for NumPy array shape (channel, time, z, y, x)
    Returns numpy array of shape (time, 2) with only z shifts
    """
    # replicate da.swapaxes and da.average
    z_movie = np.average(np.swapaxes(_data[ref_channel], 2, 1), axis=1)

    print('Determining drift in Z')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(z_movie) - 1)):
        s, e, p = phase_cross_correlation(
            z_movie[t], z_movie[t + 1], normalization=None
        )
        shifts.append(s)
        error.append(e)
        phasediff.append(p)

    shifts_z = np.cumsum(np.array(shifts), axis=0)
    # zero out y component same as Dask
    for i in shifts_z:
        i[1] = 0

    shifts_z = shifts_z.tolist()
    shifts_z.insert(0, [0, 0])
    shifts_z = np.asarray(shifts_z)

    plt.title('Z-Drift')
    plt.plot(shifts_z[:, 0], label='z')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Drift in pixel')
    plt.savefig('Z-Drift.svg')
    plt.clf()

    return shifts_z


def translate_stack_numpy(_image, _shift):
    """
    Apply 2D XY affine transform to a stack shape (channel, z, y, x)
    """
    tform = AffineTransform(translation=-_shift)
    _image_out = []
    for c, C in enumerate(_image):
        _out_stack = []
        for z, Z in enumerate(C):
            # replicate delayed call signature by using np.array(tform)
            _out_stack.append(affine_transform(Z, np.array(tform)))
        _image_out.append(np.stack(_out_stack, axis=0))
    return np.stack(_image_out, axis=0)


def apply_xy_drift_numpy(_data, xy_drift):
    """
    Apply XY drift to NumPy array shape (channel, time, z, y, x)
    Returns corrected array of same shape
    """
    # replicate da.swapaxes(_data,0,1)
    _data = np.swapaxes(_data, 0, 1)

    print('Scheduling tasks...')
    _translated_data = []
    for t, T in tqdm(list(enumerate(_data))):
        _translated_data.append(translate_stack_numpy(T, xy_drift[t]))

    _data_out = np.stack(_translated_data, axis=0)
    # replicate swapping back
    _data_out = np.swapaxes(_data_out, 0, 1)

    print('XY-shift has been scheduled.')
    return _data_out


def rotate_stack_numpy(_image, _alpha):
    """
    Apply 2D rotation to a stack shape (channel, z, y, x)
    """
    _image_out = []
    for c, C in enumerate(_image):
        _out_stack = []
        for z, Z in enumerate(C):
            _out_stack.append(rotate(Z, -_alpha, reshape=False))
        _image_out.append(np.stack(_out_stack, axis=0))
    return np.stack(_image_out, axis=0)


def apply_z_drift_numpy(_data, z_drift):
    """
    Apply Z drift to NumPy array shape (channel, time, z, y, x)
    Returns corrected array of same shape
    """
    # replicate axis swaps
    _data = np.swapaxes(_data, 0, 1)
    _data = np.swapaxes(_data, 3, 2)

    _data_out = []
    for t, T in tqdm(list(enumerate(_data))):
        _data_out.append(translate_stack_numpy(T, z_drift[t]))

    _data_out = np.stack(_data_out, axis=0)
    # swap back
    _data_out = np.swapaxes(_data_out, 2, 3)
    _data_out = np.swapaxes(_data_out, 0, 1)

    return _data_out


def crop_data_numpy(data, xy_drift, z_drift):
    """
    Crop edges based on drift to remove border artifacts
    """
    y_crop = [int(np.max(xy_drift[:, 0])), int(np.shape(data)[-1] - abs(np.min(xy_drift[:, 0])))]
    x_crop = [int(np.max(xy_drift[:, 1])), int(np.shape(data)[-2] - abs(np.min(xy_drift[:, 1])))]
    z_crop = [int(np.max(z_drift[:, 0])), int(np.shape(data)[-3] - abs(np.min(z_drift[:, 0])))]

    return data[:, :,
                z_crop[0]:z_crop[1],
                y_crop[0]:y_crop[1],
                x_crop[0]:x_crop[1]]


def get_rotation_numpy(data, ref_channel=0):
    """
    Estimate rotation drift for NumPy array shape (channel, time, z, y, x)
    Returns numpy array of shape (time,)
    """
    xy_movie = np.average(data[ref_channel], axis=1)
    radius = int(min(data.shape[3], data.shape[4]) / 2)

    print('Determining rotation')
    shifts, error, phasediff = [], [], []
    for t in tqdm(range(len(xy_movie) - 1)):
        s, e, p = phase_cross_correlation(
            warp_polar(xy_movie[t], radius=radius),
            warp_polar(xy_movie[t + 1], radius=radius),
            normalization=None
        )
        shifts.append(s)
        error.append(e)
        phasediff.append(p)

    shifts_a = np.cumsum(np.array(shifts), axis=0)
    # replicate bug: zero original shifts then ignore in cumsum
    for i in shifts:
        i[0] = 0

    shifts_a = shifts_a.tolist()
    shifts_a.insert(0, [0, 0])
    shifts_a = np.asarray(shifts_a)

    plt.title('alpha-Drift')
    plt.plot(shifts_a[:, 0], label='alpha')
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('Rotation in degree')
    plt.savefig('Rotation-Drift.svg')
    plt.clf()

    return shifts_a[:, 0]


def apply_alpha_drift_numpy(_data, alpha_drift):
    """
    Apply rotation drift to NumPy array shape (channel, time, z, y, x)
    Returns corrected array of same shape
    """
    _data = np.swapaxes(_data, 0, 1)
    _data_out = []
    for t, T in tqdm(list(enumerate(_data))):
        _data_out.append(rotate_stack_numpy(T, alpha_drift[t]))

    _data_out = np.stack(_data_out, axis=0)
    _data_out = np.swapaxes(_data_out, 0, 1)

    return _data_out
