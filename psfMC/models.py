from __future__ import division
from warnings import warn
import numpy as np
from scipy.special import gamma
from pymc import Uniform, Normal, deterministic
import time

# Pixels that have zero weight will be replaced with a very small weight
# TODO: Is there a way to use masked arrays to skip bad pixels instead?
_zero_weight = 1e-20

_show_debug_output = False

np.seterr(divide='ignore')

# components is a list of tuples. first element is type (psf, sersic)
# further elements are min and max of uniform search regions
# [('psf', 120, 136, 120, 136, 1e-14, 1e4)]
# type xmin xmax ymin ymax p1min p1max etc

_t_in = 0.0
def _debug_timer(step, name=''):
    global _t_in
    if not _show_debug_output:
        return
    if step == 'start':
        _t_in = time.time()
    elif step == 'stop':
        print '{}: {:.2e}'.format(name, time.time() - _t_in),
    else:
        print ''


def _convolve(img, kernel):
    return np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(kernel, img.shape))


def _add_sersic(arr, xy, mag, reff, index, axis_ratio, angle):
    """
    Add Sersic profile with supplied parameters to a numpy array
    """
    kappa = 1.9992*index - 0.3271
    fluxtot = 10 ** (mag / -2.5)
    sbeff = fluxtot / (2 * np.pi * reff**2 * axis_ratio * np.exp(kappa) *
                       index * np.power(kappa, -2*index) * gamma(2*index))
    angle = np.deg2rad(angle)
    sin_ang, cos_ang = np.sin(angle), np.cos(angle)
    M_inv_scale = np.diag((1/reff, 1/(reff*axis_ratio))) ** 2
    M_rot = np.asarray((cos_ang, -sin_ang, sin_ang, cos_ang)).reshape(2,2)
    M_inv_rot = np.asarray((cos_ang, sin_ang, -sin_ang, cos_ang)).reshape(2,2)
    M_inv_xform = np.dot(np.dot(M_rot, M_inv_scale), M_inv_rot)

    coords = [np.arange(arr.size) % arr.shape[1],
              np.arange(arr.size) // arr.shape[1]]
    coords = np.transpose(coords).astype(arr.dtype)
    coords -= xy
    radii = np.sqrt(np.sum(
        (coords.T * np.dot(M_inv_xform, coords.T)),
        axis=0))
    radii = radii.reshape(arr.shape)
    arr += sbeff * np.exp(-kappa * (np.power(radii, 1/index) - 1))
    return arr


def _add_point_source(arr, xy, mag):
    """
    Add point source with supplied parameters to a numpy array
    """
    flux = 10 ** (mag / -2.5)
    xint, xfrac = xy[0] // 1, xy[0] % 1
    yint, yfrac = xy[1] // 1, xy[1] % 1
    arr[yint:yint+2, xint:xint+2] += flux * np.outer((yfrac, 1-yfrac),
                                                     (xfrac, 1-xfrac))
    return arr


def multicomponent_model(subData, subDataIVM, psf, psfIVM,
                         components=[]):
    model_comps = []
    modelpx = np.zeros_like(subData)

    for count, component in enumerate(components):
        name = component[0]
        xy = np.asarray(component[1:5]).reshape((2, 2))
        if name == 'psf':
            xy = Uniform('{}_{}_xy'.format(count, name),
                         lower=np.min(xy, axis=1), upper=np.max(xy, axis=1))
            mag = Uniform('{}_{}_mag'.format(count, name),
                          lower=min(component[5:7]),
                          upper=max(component[5:7]))
            model_comps += [('psf', xy, mag)]

        elif name == 'sersic':
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


    @deterministic(plot=False)
    def raw_model(model_comps=model_comps):
        _debug_timer('start')
        # TODO: shouldn't have to make this writeable every time
        modelpx.flags.writeable = True
        modelpx[:,:] = 0
        for comp in model_comps:
            if comp[0] == 'psf':
                _add_point_source(modelpx, *comp[1:])
            elif comp[0] == 'sersic':
                _add_sersic(modelpx, *comp[1:])
            else:
                warn('Skipping unrecognized component {}'.format(comp[0]))
        _debug_timer('stop', name='Model')
        return modelpx


    @deterministic(plot=False)
    def convolved_model(psf=psf, rawmodel=raw_model):
        _debug_timer('start')
        cmodel = _convolve(rawmodel, psf)
        _debug_timer('stop', name='Convolve')
        return cmodel


    @deterministic(plot=False)
    def composite_IVM(subDataIVM=subDataIVM, psfIVM=psfIVM,
                      rawmodel=raw_model):
        _debug_timer('start')
        # TODO: Ensure math here is correct
        modelRMS = _convolve(rawmodel, 1/np.sqrt(psfIVM))
        # Set zero-weight pixels to very small number instead
        badpx = (modelRMS <= 0) | (subDataIVM <= 0)
        badpx |= ~np.isfinite(modelRMS) | ~np.isfinite(subDataIVM)
        compIVM = np.where(badpx, _zero_weight,
                           1 / (modelRMS**2 + 1 / subDataIVM))
        _debug_timer('stop', name='IVM')
        _debug_timer('final')
        return compIVM

    # TODO: Use skellam distribution instead of Normal for discrete data
    data = Normal('data', value=subData, observed=True,
                  mu=convolved_model, tau=composite_IVM)

    model_comps += [raw_model, convolved_model, composite_IVM, data]

    return model_comps
