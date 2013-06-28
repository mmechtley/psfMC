from __future__ import division
import time
import numpy as np
from math import fsum


_show_timing_info = False


_timers = dict()
def debug_timer(step, name=''):
    """
    Hacky lightweight timer, for profiling model creation
    """
    if not _show_timing_info:
        return
    if step == 'start':
        _timers[name] = time.time()
    elif step == 'stop':
        print '{}: {:.2e}'.format(name, time.time() - _timers[name]),
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
    img_pad[pad[0]//2:pad[0]//2 + img.shape[0],
            pad[1]//2:pad[1]//2 + img.shape[1]] = img
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


def mask_bad_pixels(obs_data, obs_ivm, psf_data, psf_ivm):
    """
    Sanitize input arrays by masking out bad pixels
    """
    # For observed data, we use numpy masked array to simply ignore bad pixels
    # We set them to zero in the weight map, even though they ought be already
    badpx = ~np.isfinite(obs_data) | ~np.isfinite(obs_ivm) | (obs_ivm <= 0)
    obs_data = np.ma.masked_array(obs_data, mask=badpx)
    obs_ivm[badpx] = 0
    # We don't want zero-weight pixels in the PSF to contribute to the RMS,
    # so we simply set them to 0 in both data and weight map
    badpx = ~np.isfinite(psf_data) | ~np.isfinite(psf_ivm) | (psf_ivm <= 0)
    psf_data[badpx] = 0
    psf_ivm[badpx] = 0
    return obs_data, obs_ivm, psf_data, psf_ivm


def normed_psf(psf_data, psf_ivm):
    """
    Returns normed psf and correspondingly scaled IVM.
    Uses math.fsum for its stable summation algorithm
    """
    psf_sum = fsum(psf_data.flat)
    return psf_data / psf_sum, psf_ivm * psf_sum**2