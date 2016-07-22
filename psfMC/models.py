from __future__ import division
from pymc import deterministic, Normal
from pymc.MCMC import MCMC
from numpy import zeros_like, seterr
from .array_utils import convolve
from .ModelComponents import Configuration, PSF
from .model_parser import component_list_from_file


# TODO: This should probably be a class instead
def multicomponent_model(components, **kwargs):
    """
    Multi-component model for MCMC psf fitting. Components is a list of
    ComponentBase subclasses, or a model definition python file parsable by
    model_parser.
    Returns an MCMC object ready for sampling. **kwargs are passed to MCMC()
    """
    seterr(divide='ignore')
    if not hasattr(components, '__iter__') or isinstance(components, str):
        try:
            components = component_list_from_file(components)
        except IOError as err:
            message = 'Unable to open components file {}. Does it exist?'
            err.message = message.format(components)
            raise err

    # First find the Configuration component and take it out of the list since
    # it's treated separately
    try:
        config = [comp for comp in components
                  if isinstance(comp, Configuration)].pop()
        components.remove(config)
    except (IndexError, ValueError):
        raise ValueError('Unable to find the Configuration component, required'
                         'for setting up input images.')

    for count, component in enumerate(components):
        component.update_trace_names(count=count)

    # List of only PSF components, for making PSF-subtracted images
    psf_comps = [comp for comp in components if isinstance(comp, PSF)]

    @deterministic(plot=False, trace=False)
    def raw_model(model_comps=components):
        modelpx = zeros_like(config.obs_var)
        for comp in model_comps:
            comp.add_to_array(modelpx, mag_zp=config.mag_zeropoint,
                              coords=config.coords)
        return modelpx

    @deterministic(plot=False, trace=False)
    def convolved_model(psf=config.psf_selector, raw_model=raw_model):
        return convolve(raw_model, psf.psf())

    @deterministic(plot=False, trace=False)
    def composite_ivm(obs_var=config.obs_var, psf=config.psf_selector,
                      raw_model=raw_model):
        # Model and observation variances are independent, so additive
        # FIXME: odd sizes (127x127) cause this to screw up in one dimension?
        model_var = convolve(raw_model**2, psf.variance())
        comp_ivm = 1 / (model_var + obs_var)
        return comp_ivm

    @deterministic(plot=False, trace=False)
    def residual(obs_data=config.obs_data, convolved_model=convolved_model):
        return obs_data - convolved_model

    @deterministic(plot=False, trace=False)
    def point_source_subtracted(psf_comps=psf_comps, psf=config.psf_selector):
        # Note: this is NOT called every iteration during sampling (no observed
        # stochastic depends on it). Only when generating posterior model images
        psf_px = zeros_like(config.obs_var)
        for comp in psf_comps:
            comp.add_to_array(psf_px, config.mag_zeropoint)
        psf_px = convolve(psf_px, psf.psf())
        return config.obs_data - psf_px

    data = Normal('data', value=config.obs_data[~config.bad_px],
                  mu=convolved_model[~config.bad_px],
                  tau=composite_ivm[~config.bad_px],
                  observed=True, trace=False)

    stochastics = [config.psf_selector, raw_model, convolved_model,
                   composite_ivm, residual, point_source_subtracted, data]
    stochastics += components

    mcmc = MCMC(stochastics, **kwargs)
    # FIXME: This is kind of an abuse of duck-typing. Subclass MCMC instead?
    mcmc.obs_header = config.obs_header
    return mcmc
