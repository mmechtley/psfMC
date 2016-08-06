from __future__ import division
import numpy as np
from .array_utils import convolve
from .ModelComponents import Configuration, PointSource
from .model_parser import component_list_from_file


class MultiComponentModel(object):
    """
    A composite 2D surface brightness model defined from a list of individual
    image components (Sersics, etc.) and a Configuration component. These can
    be supplied as either a list of Component objects, or a file that will be
    parsed by the component_list_from_file model parser.
    """
    def __init__(self, components):
        np.seterr(divide='ignore')

        if not hasattr(components, '__iter__') or isinstance(components, str):
            try:
                components = component_list_from_file(components)
            except IOError as err:
                message = 'Unable to open model file {}. Does it exist?'
                err.message = message.format(components)
                raise err

        # First find the Configuration component and take it out of the list
        # since it's treated separately
        try:
            config = [comp for comp in components
                      if isinstance(comp, Configuration)].pop()
            components.remove(config)
        except (IndexError, ValueError):
            raise ValueError(
                'Unable to find the Configuration component, required'
                'for setting up input images.')
        # Append the PSF selector to the component list TODO: refactor this
        components.append(config.psf_selector)

        # Set trace names and FITS header names based on total num of components
        for count, component in enumerate(components):
            component.update_stochastic_names(count=count)

        # List of only PSF components, for making PSF-subtracted images
        psf_comps = [comp for comp in components
                     if isinstance(comp, PointSource)]

        # List of only components that are used to generate raw image
        raw_img_comps = [comp for comp in components
                         if hasattr(comp, 'add_to_array')]

        self.config = config
        self.components = components
        self.raw_model_components = raw_img_comps
        self.psf_comps = psf_comps
        self.obs_header = self.config.obs_header

        self._param_vector = np.zeros(self.num_params)

    def init_params_from_priors(self, nwalkers):
        """
        Generate an array of initial parameter values (walker starting
        locations) by drawing randomly from the priors.

        :param nwalkers: Number of parameter vectors to generate
        :return: numpy array of shape (nwalkers, nparams)
        """
        initial_params = np.zeros((nwalkers, self.num_params))
        for walker in range(nwalkers):
            param_vec = np.array([])
            for comp in self.components:
                comp_stochs = comp.set_stochastic_values()
                param_vec = np.concatenate((param_vec, comp_stochs))
            initial_params[walker, :] = param_vec
        return initial_params

    @property
    def num_params(self):
        """
        Total number of stochastic parameters in the model
        """
        return np.sum([comp.num_stochastics() for comp in self.components])

    @property
    def param_names(self):
        """
        List of all stochastic parameter names, ordered as in the vector emcee
        supplies to the posterior function.
        """
        return np.sum([comp.stochastic_names() for comp in self.components])

    @property
    def param_lens(self):
        """
        Length of all stochastic parameter values, ordered as in the vector
        emcee supplies to the posterior function.
        """
        return np.sum([comp.stochastic_lens() for comp in self.components])

    @property
    def param_values(self):
        """
        Current value of all parameters, as a dict. Mostly for debug inspection
        """
        split_indexes = np.cumsum(self.param_lens)
        split_values = np.split(self._param_vector, split_indexes)
        param_dict = dict(zip(self.param_names, split_values))
        return param_dict

    @param_values.setter
    def param_values(self, value_vector):
        """
        Store parameter vector, then call value setter on all child components
        """
        self._param_vector = value_vector
        pnum_start = 0
        for comp in self.components:
            n_params = comp.num_stochastics()
            comp_params = value_vector[pnum_start:pnum_start + n_params]
            comp.set_stochastic_values(comp_params)
            pnum_start += n_params

    def log_priors(self):
        """
        Joint log-probability of priors for all stochastic parameters
        """
        return np.sum([comp.log_priors() for comp in self.components])

    @staticmethod
    def log_posterior(param_values, **kwargs):
        """
        log-probability of the posterior distribution for a given set of
        parameters. Note: because of how emcee calls this, the posterior
        function is static. The MultiComponentModel object must be passed using
        kwargs={'model': mc_model} in the Sampler constructor.

        :param param_values: Vector of values for all stochastic parameters
        :param kwargs: MUST include 'model' kwarg
        :return: log-likelihood + joint log-prior probability
        """
        model = kwargs.pop('model')
        model.param_values = param_values

        # Calculate prior, and early out for unsupported prior values
        log_priors = model.log_priors()
        if not np.isfinite(log_priors):
            return -np.inf

        raw_px = model.raw_model()
        conv_px = model.convolved_model(raw_px)
        ivm_flat = model.composite_ivm(raw_px)[~model.config.bad_px]
        resid_flat = model.residual(conv_px)[~model.config.bad_px]

        # This is Normal log-likelihood. Consider letting the user choose Normal
        # or Poisson (or maybe others). However, just writing rather than using
        # a distributions.Normal object makes this function about 20% faster
        log_likelihood = -0.5 * np.sum(resid_flat**2 * ivm_flat
                                       - np.log(0.5 / np.pi * ivm_flat))

        # FIXME: kinda a hack. log-likelihood is NaN sometimes, find out why
        if not np.isfinite(log_likelihood):
            return -np.inf

        return log_likelihood + log_priors

    def raw_model(self):
        """
        Generates raw model image (before convolution with the PSF)
        """
        modelpx = np.zeros_like(self.config.obs_var)
        for comp in self.raw_model_components:
            comp.add_to_array(modelpx, mag_zp=self.config.mag_zeropoint,
                              coords=self.config.coords)
        return modelpx

    def convolved_model(self, raw_px=None):
        """
        Convolves raw model image with the PSF

        :param raw_px: Optional pre-computed raw model image
        """
        if raw_px is None:
            raw_px = self.raw_model()
        return convolve(raw_px, self.config.psf_selector.psf)

    def composite_ivm(self, raw_px=None):
        """
        Generates composite inverse variance map, i.e. IVM that contains
        contributions from both the original data, and the PSF convolved with
        the model

        :param raw_px: Optional pre-computed raw model image
        """
        if raw_px is None:
            raw_px = self.raw_model()
        # Model and observation variances are independent, so additive
        # FIXME: odd sizes (127x127) cause this to screw up in one dimension?
        model_var = convolve(raw_px ** 2,
                             self.config.psf_selector.variance)
        comp_ivm = 1 / (model_var + self.config.obs_var)
        return comp_ivm

    def residual(self, convolved_px=None, raw_px=None):
        """
        Generates residual image, i.e. data - model

        :param convolved_px: Optional pre-computed PSF-convolved model
        :param raw_px: If convolved_px is not supplied, optional pre-computed
        raw model image
        """
        if convolved_px is None:
            if raw_px is None:
                raw_px = self.raw_model()
            convolved_px = self.convolved_model(raw_px)
        return self.config.obs_data - convolved_px

    def point_source_subtracted(self):
        """
        Generates image with only point sources subtracted. Note: this is NOT
        called every iteration during sampling, only when generating posterior
        model images
        """
        psf_px = np.zeros_like(self.config.obs_var)
        for comp in self.psf_comps:
            comp.add_to_array(psf_px, self.config.mag_zeropoint)
        psf_px = convolve(psf_px, self.config.psf_selector.psf)
        return self.config.obs_data - psf_px
