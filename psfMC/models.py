from __future__ import division
import numpy as np
from pymc import deterministic, Normal
from pymc.Model import Model
from .array_utils import *
from .ModelComponents import Sky
from .model_parser import component_list_from_file
# import matplotlib.pyplot as pp

# TODO: Is there a way to use masked arrays to skip bad pixels instead?
# Pixels that have zero weight will be replaced with a very small weight
_zero_weight = 1e-20


def multicomponent_model(obs_data, obs_ivm, psf_data, psf_ivm,
                         components=None, mag_zp=0):
    """
    Multi-component model for MCMC psf fitting. Components is a list of
    ComponentBase subclasses, or a model definition python file parsable by
    model_parser
    """
    if isinstance(components, basestring):
        try:
            components = component_list_from_file(components)
        except IOError, err:
            message = 'Unable to open components file {}. Does it exist?'
            err.message = message.format(components)
            raise err

    np.seterr(divide='ignore')

    # Normalize the PSF kernel
    psf_data, psf_ivm = normed_psf(psf_data, psf_ivm)

    psf_rms = np.where(psf_ivm == 0, 0, 1 / np.sqrt(psf_ivm))

    # pad the psf arrays to the same size as the data, for fft
    f_psf = pad_and_rfft_image(psf_data, obs_data.shape)
    f_psf_rms = pad_and_rfft_image(psf_rms, obs_data.shape)

    # pre-compute data x,y coordinates
    data_coords = array_coords(obs_data)

    model_comps = []
    stochastics = []
    sky = Sky(adu=0.0)

    for count, component in enumerate(components):
        component.update_trace_names(count=count)

        # Sky is added after convolution
        if component.__class__.__name__ == 'Sky':
            sky = component
            stochastics += [sky]
        else:
            model_comps += [component]

    @deterministic(plot=False, trace=False)
    def raw_model(model_comps=model_comps):
        modelpx = np.zeros_like(obs_data)
        for comp in model_comps:
            debug_timer('start')
            comp.add_to_array(modelpx, mag_zp, coords=data_coords)
            debug_timer('stop', name=comp.__class__.__name__)
        return modelpx

    @deterministic(plot=False, trace=False)
    def convolved_model(f_psf=f_psf, raw_model=raw_model):
        debug_timer('start')
        cmodel = convolve(raw_model, f_psf)
        debug_timer('stop', name='Convolve')
        return cmodel

    @deterministic(plot=False, trace=False)
    def composite_ivm(obs_ivm=obs_ivm, f_psf_rms=f_psf_rms,
                      raw_model=raw_model):
        debug_timer('start')
        # f * (g + h) = (f * g) + (f * h), so convolve PSF RMS map with model to
        # get model RMS map
        # TODO: should this be modelIVM = convolve(rawmodel**2, f_psf_var)?
        modelRMS = convolve(raw_model, f_psf_rms)
        # Set zero-weight pixels to very small number instead
        badpx = (modelRMS <= 0) | (obs_ivm <= 0)
        compIVM = np.where(badpx, _zero_weight, 1 / (modelRMS**2 + 1 / obs_ivm))
        # for arr in (obs_data.mask, raw_model, modelRMS, compIVM):
        #     pp.imshow(np.log10(arr), interpolation='nearest')
        #     pp.colorbar()
        #     pp.show()
        # exit(1)
        debug_timer('stop', name='IVM')
        debug_timer('final')
        return compIVM

    data = Normal('data', value=obs_data, mu=convolved_model+sky.adu,
                  tau=composite_ivm, observed=True, trace=False)

    stochastics += [raw_model, convolved_model, composite_ivm, data]
    stochastics += model_comps

    return Model(stochastics)
