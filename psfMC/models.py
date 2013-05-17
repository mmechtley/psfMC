from __future__ import division
import time
import numpy as np
from warnings import warn
from scipy.special import gamma
from pymc import Uniform, Normal, deterministic
import matplotlib.pyplot as pp

# TODO: Is there a way to use masked arrays to skip bad pixels instead?
# Pixels that have zero weight will be replaced with a very small weight
_zero_weight = 1e-20

_show_timing_info = False

_t_in = 0.0


def _debug_timer(step, name=''):
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


def _pad_and_rfft_image(img, newshape):
    """
    Pads the psf array to the size described by imgshape, then run rfft to put
    it in Fourier space.
    """
    # TODO: pad with white noise instead of zeros?
    pad = np.asarray(newshape) - np.asarray(img.shape)
    if np.any(pad < 0):
        raise ValueError('PSF image size must be smaller than data image size')
    img_pad = np.zeros(newshape, dtype=img.dtype)
    img_pad[pad[0]//2:pad[0]//2+img.shape[0],
            pad[1]//2:pad[1]//2+img.shape[1]] = img
    return np.fft.rfft2(img_pad)


def _convolve(img, fourier_kernel):
    """
    FFT-based convolution, using the Convolution Theorem. This is about 100x
    faster than using scipy.ndimage.convolve, due to FFT. But it effectively
    forces the boundary mode to be wrap. The kernel is supplied pre-computed
    """
    # TODO: consider padding to power-of-two for extra speed
    return np.fft.ifftshift(np.fft.irfft2(np.fft.rfft2(img) * fourier_kernel))


def _coord_array(arr):
    """
    Returns arr.size x 2 array of x, y coordinate for each cell in arr
    """
    coords = [np.arange(arr.size) % arr.shape[1],
              np.arange(arr.size) // arr.shape[1]]
    return np.transpose(coords).astype(arr.dtype)


def add_sersic(arr, magZP, xy, mag, reff, index, axis_ratio, angle,
               coords=None):
    """
    Add Sersic profile with supplied parameters to a numpy array. Array is
    assumed to be in counts per second, ie the brightness of a pixel is
    m = -2.5*log(pixel value) + magZP

    :param arr: Numpy array to add sersic profile to
    :param magZP: Magnitude zeropoint (i.e. magnitude of 1 count/second)
    :param xy: Numpy array, placement in pixels within the array
    :param mag: Integrated magnitude of profile
    :param reff: Effective radius, in pixels
    :param index: Sersic index n. 0.5=gaussian, 1=exponential, 4=de Vaucouleurs
    :param axis_ratio: Ratio of minor over major axis
    :param angle: Position angle of major axis, degrees clockwise of right
    :param coords:
    """
    if coords is None:
        coords = _coord_array(arr)
    else:
        coords = coords.copy()

    kappa = 1.9992*index - 0.3271
    fluxtot = 10 ** ((mag - magZP) / -2.5)
    sbeff = fluxtot / (2 * np.pi * reff**2 * axis_ratio * np.exp(kappa) *
                       index * np.power(kappa, -2*index) * gamma(2*index))
    angle = np.deg2rad(angle)
    sin_ang, cos_ang = np.sin(angle), np.cos(angle)

    # Matrix representation of ellipse: http://en.wikipedia.org/wiki/Ellipsoid
    M_inv_scale = np.diag((1/reff, 1/(reff*axis_ratio))) ** 2
    M_rot = np.asarray((cos_ang, -sin_ang, sin_ang, cos_ang)).reshape(2, 2)
    M_inv_xform = np.dot(np.dot(M_rot, M_inv_scale), M_rot.T)

    coords -= xy
    radii = np.sqrt(np.sum(
        (coords.T * np.dot(M_inv_xform, coords.T)),
        axis=0))
    radii = radii.reshape(arr.shape)
    arr += sbeff * np.exp(-kappa * (np.power(radii, 1/index) - 1))
    return arr, sbeff


def add_point_source(arr, magZP, xy, mag):
    """
    Add point source with supplied parameters to a numpy array. Array is
    assumed to be in counts per second, ie the brightness of a pixel is
    m = -2.5*log(pixel value) + magZP
    Linearly interpolation is used for subpixel positions.

    :param arr: Numpy array to add psf to
    :param magZP: Magnitude zeropoint (i.e. magnitude of 1 count/second)
    :param xy: Numpy array, placement in pixels within the array.
    :param mag: Integrated magnitude of point source
    """
    flux = 10 ** ((mag - magZP) / -2.5)
    xint, xfrac = xy[0] // 1, xy[0] % 1
    yint, yfrac = xy[1] // 1, xy[1] % 1
    arr[yint:yint+2, xint:xint+2] += flux * np.outer((yfrac, 1-yfrac),
                                                     (xfrac, 1-xfrac))
    return arr


def multicomponent_model(obs_data, obs_ivm, psf_data, psf_ivm,
                         components=None, mag_zp=0):
    """
    Multi-component model for MCMC psf fitting.
    Components is a list of tuples. First element is type (psf, sersic).
    Further elements are min and max of uniform search regions, e.g.
    type xmin xmax ymin ymax p1min p1max etc
    [('psf', 120, 136, 120, 136, 17, 20),
     ('sersic', 120, 136, 120, 136, 21, 28, 1.5, 3.5, 0.5, 8, 0.1, 1.0, 0, 360)]
    """
    np.seterr(divide='ignore')

    psf_rms = np.where(psf_ivm == 0, 0, 1 / np.sqrt(psf_ivm))

    # pad the psf arrays to the same size as the data, for fft
    f_psf = _pad_and_rfft_image(psf_data, obs_data.shape)
    f_psf_rms = _pad_and_rfft_image(psf_rms, obs_data.shape)

    # pre-compute data x,y coordinates
    data_coords = _coord_array(obs_data)

    model_comps = []
    stochastics = []
    sky = 0.0

    for count, component in enumerate(components):
        name = component[0]
        if name == 'sky':
            sky = Uniform('{}_{}_adu'.format(count, name),
                              lower=min(component[1:3]),
                              upper=max(component[1:3]))
            stochastics += [sky]
        elif name == 'psf':
            xy = np.asarray(component[1:5]).reshape((2, 2))
            xy = Uniform('{}_{}_xy'.format(count, name),
                         lower=np.min(xy, axis=1), upper=np.max(xy, axis=1))
            mag = Uniform('{}_{}_mag'.format(count, name),
                          lower=min(component[5:7]),
                          upper=max(component[5:7]))
            model_comps += [('psf', xy, mag)]

        elif name == 'sersic':
            xy = np.asarray(component[1:5]).reshape((2, 2))
            xy = Uniform('{}_{}_xy'.format(count, name),
                         lower=np.min(xy, axis=1), upper=np.max(xy, axis=1))
            mag = Uniform('{}_{}_mag'.format(count, name),
                          lower=min(component[5:7]),
                          upper=max(component[5:7]))
            rad_eff = Uniform('{}_{}_re'.format(count, name),
                              lower=min(component[7:9]),
                              upper=max(component[7:9]))
            sersic_n = Uniform('{}_{}_n'.format(count, name),
                               lower=min(component[9:11]),
                               upper=max(component[9:11]))
            axis_ratio = Uniform('{}_{}_axisratio'.format(count, name),
                                 lower=min(component[11:13]),
                                 upper=max(component[11:13]))
            angle = Uniform('{}_{}_angle'.format(count, name),
                            lower=min(component[13:15]),
                            upper=max(component[13:15]))

            model_comps += [('sersic', xy, mag, rad_eff, sersic_n,
                            axis_ratio, angle)]
        else:
            warn('Unrecognized component: {}'.format(name))


    @deterministic(plot=False, trace=False)
    def raw_model(model_comps=model_comps):
        _debug_timer('start')
        modelpx = np.zeros_like(obs_data)
        for comp in model_comps:
            if comp[0] == 'psf':
                add_point_source(modelpx, mag_zp, *comp[1:])
            elif comp[0] == 'sersic':
                add_sersic(modelpx, mag_zp, *comp[1:], coords=data_coords)
        _debug_timer('stop', name='Model')
        return modelpx


    @deterministic(plot=False, trace=False)
    def convolved_model(f_psf=f_psf, raw_model=raw_model):
        _debug_timer('start')
        cmodel = _convolve(raw_model, f_psf)
        _debug_timer('stop', name='Convolve')
        return cmodel


    @deterministic(plot=False, trace=False)
    def composite_ivm(obs_ivm=obs_ivm, f_psf_rms=f_psf_rms,
                      raw_model=raw_model):
        _debug_timer('start')
        # f * (g + h) = (f * g) + (f * h), so convolve PSF RMS map with model to
        # get model RMS map
        modelRMS = _convolve(raw_model, f_psf_rms)
        # Set zero-weight pixels to very small number instead
        badpx = (modelRMS <= 0) | (obs_ivm <= 0)
        compIVM = np.where(badpx, _zero_weight, 1 / (modelRMS**2 + 1 / obs_ivm))
        # for arr in (raw_model, modelRMS, compIVM):
        #     pp.imshow(np.log10(arr), interpolation='nearest')
        #     pp.colorbar()
        #     pp.show()
        # exit(1)
        _debug_timer('stop', name='IVM')
        _debug_timer('final')
        return compIVM

    data = Normal('data', value=obs_data, mu=convolved_model+sky,
                  tau=composite_ivm, observed=True, trace=False)

    stochastics += [raw_model, convolved_model, composite_ivm, data]
    stochastics += model_comps

    return stochastics
