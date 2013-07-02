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
    psf_var = np.where(psf_ivm == 0, 0, 1 / psf_ivm)

    # pad the psf arrays to the same size as the data, for fft
    f_psf = pad_and_rfft_image(psf_data, obs_data.shape)
    f_psf_var = pad_and_rfft_image(psf_var, obs_data.shape)

    # pre-compute variance map and bad pixel mask for observation
    obs_mask = obs_ivm <= 0
    obs_var = np.where(obs_mask, 0, 1 / obs_ivm)

    # pre-compute data x,y coordinates
    data_coords = array_coords(obs_data)

    model_comps = []
    stochastics = []
    sky = Sky(adu=0.0)

    for count, component in enumerate(components):
        component.update_trace_names(count=count)

        # Sky is added after convolution
        if isinstance(component, Sky):
            sky = component
            stochastics += [sky]
        else:
            model_comps += [component]

    @deterministic(plot=False, trace=False)
    def raw_model(model_comps=model_comps):
        modelpx = np.zeros_like(obs_data)
        for comp in model_comps:
            comp.add_to_array(modelpx, mag_zp, coords=data_coords)
        return modelpx

    @deterministic(plot=False, trace=False)
    def convolved_model(f_psf=f_psf, raw_model=raw_model, sky=sky):
        cmodel = convolve(raw_model, f_psf)
        return cmodel + sky.adu

    @deterministic(plot=False, trace=False)
    def composite_ivm(obs_var=obs_var, obs_mask=obs_mask, f_psf_var=f_psf_var,
                      raw_model=raw_model):
        # compute model variance
        model_var = convolve(raw_model**2, f_psf_var)
        # Set zero-weight pixels to very small number instead
        badpx = (model_var <= 0) | obs_mask
        compIVM = np.where(badpx, _zero_weight, 1 / (model_var + obs_var))
        return compIVM

    @deterministic(plot=False, trace=False)
    def residual(obs_data=obs_data, convolved_model=convolved_model):
        return obs_data - convolved_model

    data = Normal('data', value=obs_data, mu=convolved_model,
                  tau=composite_ivm, observed=True, trace=False)

    stochastics += [raw_model, convolved_model, composite_ivm, residual, data]
    stochastics += model_comps

    return Model(stochastics)
