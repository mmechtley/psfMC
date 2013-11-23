from __future__ import division
from pymc import deterministic, Normal
from pymc.MCMC import MCMC
from .array_utils import *
from .ModelComponents import Sky, PSF
from .ModelComponents.PSFSelector import PSFSelector
from .model_parser import component_list_from_file


def multicomponent_model(obs_data, obs_ivm, psf_data, psf_ivm,
                         components=None, mag_zp=0, mask_file=None, **kwargs):
    """
    Multi-component model for MCMC psf fitting. Components is a list of
    ComponentBase subclasses, or a model definition python file parsable by
    model_parser.
    Returns an MCMC object ready for sampling. **kwargs are passed to MCMC()
    """
    if isinstance(components, basestring):
        try:
            components = component_list_from_file(components)
        except IOError, err:
            message = 'Unable to open components file {}. Does it exist?'
            err.message = message.format(components)
            raise err

    np.seterr(divide='ignore')

    obs_data, obs_var, bad_px = preprocess_obs(obs_data, obs_ivm, mask_file)

    # Read in PSFs, set up selector
    if isinstance(psf_data, basestring) or isinstance(psf_ivm, basestring):
        psf_data, psf_ivm = [psf_data], [psf_ivm]
    psf_list, var_list = [], []
    for psf, ivm in zip(psf_data, psf_ivm):
        f_psf, f_psf_var = preprocess_psf(psf, ivm, obs_data.shape)
        psf_list.append(f_psf)
        var_list.append(f_psf_var)
    psf_select = PSFSelector(psf_list, var_list, filenames=psf_data)

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
        modelpx = np.zeros_like(obs_var)
        for comp in model_comps:
            comp.add_to_array(modelpx, mag_zp, coords=data_coords)
        return modelpx

    @deterministic(plot=False, trace=False)
    def convolved_model(psf=psf_select, raw_model=raw_model, sky=sky):
        cmodel = convolve(raw_model, psf.psf())
        return cmodel + sky.adu

    @deterministic(plot=False, trace=False)
    def composite_ivm(obs_var=obs_var, psf=psf_select, raw_model=raw_model):
        # Model and observation variances are independent, so additive
        # FIXME: odd sizes (127x127) cause this to screw up in one dimension?
        model_var = convolve(raw_model**2, psf.variance())
        comp_ivm = 1 / (model_var + obs_var)
        return comp_ivm

    @deterministic(plot=False, trace=False)
    def residual(obs_data=obs_data, convolved_model=convolved_model):
        return obs_data - convolved_model

    @deterministic(plot=False, trace=False)
    def point_source_subtracted(model_comps=model_comps, psf=psf_select):
        psf_px = np.zeros_like(obs_var)
        psf_comps = [comp for comp in model_comps if isinstance(comp, PSF)]
        for comp in psf_comps:
            comp.add_to_array(psf_px, mag_zp)
        psf_px = convolve(psf_px, psf.psf())
        return obs_data - psf_px

    data = Normal('data', value=obs_data[~bad_px],
                  mu=convolved_model[~bad_px],
                  tau=composite_ivm[~bad_px],
                  observed=True, trace=False)

    stochastics += [raw_model, convolved_model, composite_ivm, residual,
                    point_source_subtracted, data, psf_select]
    stochastics += model_comps

    return MCMC(stochastics, **kwargs)
