from __future__ import division
import time
import pyfits
import numpy as np
from math import fsum
from warnings import warn


_timers = dict()

_bad_px_value = 0


def debug_timer(step, name=''):
    """
    Hacky lightweight timer, for profiling model creation
    """
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
        raise NotImplementedError('PSF images larger than observation ' +
                                  'images are not yet supported')
    img_pad = np.zeros(newshape, dtype=img.dtype)
    img_pad[pad[0]//2:pad[0]//2 + img.shape[0],
            pad[1]//2:pad[1]//2 + img.shape[1]] = img
    return np.fft.rfft2(img_pad)


def convolve(img, fourier_kernel):
    """
    FFT-based convolution, using the Convolution Theorem. This is about 100x
    faster than using scipy.ndimage.convolve, due to FFT. But it effectively
    forces the boundary mode to be wrap. The kernel is supplied pre-computed.
    For improved speed, supply power-of-two arrays
    """
    return np.fft.ifftshift(np.fft.irfft2(np.fft.rfft2(img) * fourier_kernel))


def array_coords(arr):
    """
    Returns arr.size x 2 array of x, y coordinates for each cell in arr
    """
    coords = [np.arange(arr.size) % arr.shape[1],
              np.arange(arr.size) // arr.shape[1]]
    return np.transpose(coords).astype(arr.dtype)


def norm_psf(psf_data, psf_ivm):
    """
    Returns normed psf and correspondingly scaled IVM.
    Uses math.fsum for its stable summation algorithm
    """
    psf_sum = fsum(psf_data.flat)
    return psf_data / psf_sum, psf_ivm * psf_sum**2


def preprocess_obs(obs_data, obs_ivm, mask_reg=None):
    """
    Opens data and weight maps for both observations and PSF, masks out bad
    pixels, and normalizes the PSF for convolution
    """
    # Read in arrays, mask bad pixels
    obs_hdr = pyfits.getheader(obs_data)
    obs_data = pyfits.getdata(obs_data, ignore_missing_end=True)
    obs_ivm = pyfits.getdata(obs_ivm, ignore_missing_end=True)

    # Generate bad pixel mask. Bad pixels get 0 weight in weight map, and are
    # excluded from fitting
    badpx = ~np.isfinite(obs_data) | ~np.isfinite(obs_ivm) | (obs_ivm <= 0)

    # Pre-compute variance map for observation. Bad pixels are given infinite
    # variance, since, when making the weight map, 1 / inf = 0.0
    obs_var = np.where(badpx, np.inf, 1 / obs_ivm)

    # Add masking regions to bad pixel mask. We leave their variance alone, to
    # facilitate photometry later
    if mask_reg is not None:
        try:
            import pyregion as preg
            regfilt = preg.open(mask_reg).as_imagecoord(obs_hdr).get_filter()
            exclude_px = ~regfilt.mask(obs_data.shape)
            badpx |= exclude_px
        except ImportError:
            warn('pyregion module could not be imported. Mask regions will ' +
                 'be ignored.')

    return obs_data, obs_var, badpx


def preprocess_psf(psf_data, psf_ivm, pad_to_shape=None):
    """
    Read in a PSF & IVM, mask bad pixels, normalize kernel, and pre-FFT
    """
    psf_data = pyfits.getdata(psf_data, ignore_missing_end=True)
    psf_ivm = pyfits.getdata(psf_ivm, ignore_missing_end=True)

    # We don't want zero-weight pixels in the PSF to contribute infinitely to
    # the variance, so we simply set them to 0 in both data and weight map
    badpx = ~np.isfinite(psf_data) | ~np.isfinite(psf_ivm) | (psf_ivm <= 0)
    psf_data[badpx] = 0
    psf_ivm[badpx] = 0

    # Normalize the PSF kernel
    psf_data, psf_ivm = norm_psf(psf_data, psf_ivm)

    # pad the psf arrays to the same size as the data, precompute fft
    psf_var = np.where(psf_ivm <= 0, 0, 1 / psf_ivm)
    f_psf = pad_and_rfft_image(psf_data, pad_to_shape)
    f_psf_var = pad_and_rfft_image(psf_var, pad_to_shape)

    return f_psf, f_psf_var