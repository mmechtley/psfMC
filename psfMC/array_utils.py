from __future__ import division
import numpy as np
from astropy.io import fits
from math import fsum
from warnings import warn


# TODO: Investigate using scipy.fftpack instead?
def pad_and_rfft_image(img, newshape):
    """
    Pads the psf array to the size described by imgshape, then run rfft to put
    it in Fourier space.
    """
    # TODO: pad with white noise instead of zeros?
    pad = np.asarray(newshape) - np.asarray(img.shape)
    if np.any(pad < 0):
        raise NotImplementedError('PSF images larger than observation images '
                                  'are not yet supported')
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


def array_coords(shape=(1, 1)):
    """
    Returns prod(shape) x 2 array of x, y coordinates for each cell in array
    with dimensions like shape
    """
    indexes = np.arange(np.product(shape))
    coords = [indexes % shape[1], indexes // shape[1]]
    return np.transpose(coords).astype('float64')


def norm_psf(psf_data, psf_ivm):
    """
    Returns normed psf and correspondingly scaled IVM.
    Uses math.fsum for its stable summation algorithm
    """
    psf_sum = fsum(psf_data.flat)
    return psf_data / psf_sum, psf_ivm * psf_sum**2


def preprocess_obs(obs_data, obs_ivm, mask_file=None):
    """
    Opens data and weight maps for both observations and PSF, masks out bad
    pixels, and normalizes the PSF for convolution
    """
    # Read in arrays, mask bad pixels
    obs_hdr = fits.getheader(obs_data)
    obs_data = fits.getdata(obs_data, ignore_missing_end=True)
    obs_ivm = fits.getdata(obs_ivm, ignore_missing_end=True)

    # Generate bad pixel mask. Bad pixels get 0 weight in weight map, and are
    # excluded from fitting
    badpx = ~np.isfinite(obs_data) | ~np.isfinite(obs_ivm) | (obs_ivm <= 0)

    # Pre-compute variance map for observation. Bad pixels are given infinite
    # variance, since, when making the weight map, 1 / inf = 0.0
    obs_var = np.where(badpx, np.inf, 1 / obs_ivm)

    # Add masking regions to bad pixel mask. We leave their variance alone, to
    # facilitate photometry later
    if mask_file is not None:
        exclude_px = mask_from_file(mask_file, obs_hdr, obs_data.shape)
        if exclude_px is not None:
            badpx |= exclude_px

    return obs_hdr, obs_data, obs_var, badpx


def mask_from_file(mask_file, obs_hdr, shape):
    """
    Create bad pixel mask from a file. File can be supplied in fits format
    (nonzero pixels denoting exclusion), or in ds9 region format.
    """
    try:
        return fits.getdata(mask_file).astype(bool)
    except IOError:
        pass  # When not in fits format

    try:
        import pyregion
        regfilt = pyregion.open(mask_file).as_imagecoord(obs_hdr).get_filter()
        return ~regfilt.mask(shape)
    except ImportError:
        pyregion = None
        warn('pyregion module could not be imported. ds9 region format masks '
             'will be ignored.')
    except UnicodeDecodeError:
        pass  # When not ds9 region format

    return None


def preprocess_psf(psf_data, psf_ivm):
    """
    Read in a PSF & IVM, mask bad pixels, normalize kernel
    Return the normed data and a corresponding (non-inverse) variance map
    """
    psf_data = fits.getdata(psf_data, ignore_missing_end=True)
    psf_ivm = fits.getdata(psf_ivm, ignore_missing_end=True)

    # We don't want zero-weight pixels in the PSF to contribute infinitely to
    # the variance, so we simply set them to 0 in both data and weight map
    badpx = ~np.isfinite(psf_data) | ~np.isfinite(psf_ivm) | (psf_ivm <= 0)
    psf_data[badpx] = 0
    psf_ivm[badpx] = 0

    # Normalize the PSF kernel, then return data and variance map
    psf_data, psf_ivm = norm_psf(psf_data, psf_ivm)
    psf_var = np.where(psf_ivm <= 0, 0, 1 / psf_ivm)
    return psf_data, psf_var


def pre_fft_psf(psf_data, psf_var, pad_to_shape=None):
    """
    Pre-compute (real) Fourier transforms of input PSFs and their variance maps,
    padding to the given size if needed
    """
    f_psf = pad_and_rfft_image(psf_data, pad_to_shape)
    f_psf_var = pad_and_rfft_image(psf_var, pad_to_shape)
    return f_psf, f_psf_var


def calculate_psf_variability(psf_data, psf_vars, debug_psfs=False):
    """
    Take a set of normalized PSFs and their corresponding variance maps, measure
    the inter-PSF (i.e. PSF variability/mismatch) variance map, and propagate
    its contribution into the individual variance maps.
    """
    if len(psf_data) == 1:
        return psf_data, psf_vars
    mismatch_var = np.var(psf_data, axis=0)

    if debug_psfs:
        meanpsf = np.mean(psf_data, axis=0)
        fits.writeto('xx_rms.fits', np.sqrt(mismatch_var))
        fits.writeto('xx_mean.fits', meanpsf)
        for num, (psf, var) in enumerate(zip(psf_data, psf_vars)):
            fits.writeto('xx_psf{:d}.fits'.format(num), psf - meanpsf)
            fits.writeto('xx_rms{:d}.fits'.format(num), np.sqrt(var))
        exit(1)

    # Add contribution of PSF mismatch to all individual variance maps
    psf_vars = [var + mismatch_var for var in psf_vars]
    return psf_data, psf_vars
