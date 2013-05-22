from __future__ import division
import time
import numpy as np


_show_timing_info = False
_t_in = 0.0


def debug_timer(step, name=''):
    """
    Hacky lightweight timer, for profiling model creation
    """
    global _t_in
    if not _show_timing_info:
        return
    if step == 'start':
        _t_in = time.time()
    elif step == 'stop':
        print '{}: {:.2e}'.format(name, time.time() - _t_in),
    else:
        print ''


def pad_and_rfft_image(img, newshape):
    """
    Pads the psf array to the size described by imgshape, then run rfft to put
    it in Fourier space.
    """
    # TODO: pad with white noise instead of zeros?
    pad = np.asarray(newshape) - np.asarray(img.shape)
    if np.any(pad < 0):
        raise ValueError('PSF image size cannot be larger than observation ' +
                         'image size')
    img_pad = np.zeros(newshape, dtype=img.dtype)
    img_pad[pad[0]//2:pad[0]//2+img.shape[0],
            pad[1]//2:pad[1]//2+img.shape[1]] = img
    return np.fft.rfft2(img_pad)


def convolve(img, fourier_kernel):
    """
    FFT-based convolution, using the Convolution Theorem. This is about 100x
    faster than using scipy.ndimage.convolve, due to FFT. But it effectively
    forces the boundary mode to be wrap. The kernel is supplied pre-computed
    """
    # TODO: consider padding to power-of-two for extra speed
    return np.fft.ifftshift(np.fft.irfft2(np.fft.rfft2(img) * fourier_kernel))


def array_coords(arr):
    """
    Returns arr.size x 2 array of x, y coordinates for each cell in arr
    """
    coords = [np.arange(arr.size) % arr.shape[1],
              np.arange(arr.size) // arr.shape[1]]
    return np.transpose(coords).astype(arr.dtype)