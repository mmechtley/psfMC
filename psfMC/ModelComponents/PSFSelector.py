import six
from .ComponentBase import ComponentBase, StochasticProperty
from ..distributions import DiscreteUniform
from ..array_utils import preprocess_psf, calculate_psf_variability, pre_fft_psf


class PSFSelector(ComponentBase):
    """
    Selects a PSF from a supplied list. This is not a public-facing class in
    ModelComponents. Rather, the Configuration component automatically
    generates a PSFSelector instance for the supplied list of PSFs when setting
    up the model.
    """
    psf_index = StochasticProperty('psf_index')

    def __init__(self, psf_list, ivm_list, data_shape):
        super(PSFSelector, self).__init__()
        # List-ify psflist and ivmlist if they are single strings

        if isinstance(psf_list, six.string_types):
            psf_list = [psf_list]
        if isinstance(ivm_list, six.string_types):
            ivm_list = [ivm_list]
        if len(psf_list) != len(ivm_list):
            raise ValueError('PSF and IVM lists must be the same length')

        if len(psf_list) > 1:
            psf_index = DiscreteUniform(low=0, high=len(psf_list))
        else:
            psf_index = 0

        # Handle PSF bad pixels, normalize
        data_var_pairs = [preprocess_psf(psf, ivm) for psf, ivm
                          in zip(psf_list, ivm_list)]
        # Calculate error contribution from mismatch (PSF variability)
        data_var_lists = calculate_psf_variability(*zip(*data_var_pairs))
        # Pre-FFT all psf models to save on per-sample computation
        f_psflist, f_varlist = zip(*[pre_fft_psf(psf, var, data_shape)
                                   for psf, var in zip(*data_var_lists)])
        self.filenames = psf_list
        self.psf_index = psf_index
        self.psf_list = f_psflist
        self.var_list = f_varlist

    def update_stochastic_names(self, count=None):
        """
        Set names for traces. Since there is only one PSFSelector, don't use
        count like in normal components
        """
        if 'psf_index' in self._priors:
            self._priors['psf_index'].name = 'PSF_Index'
            self._priors['psf_index'].fitsname = 'PSF_IDX'

    @property
    def psf(self):
        """
        Current PSF
        """
        return self.psf_list[self.psf_index]

    @property
    def variance(self):
        """
        Variance map for current PSF
        """
        return self.var_list[self.psf_index]

    @property
    def filename(self):
        """
        Filename of current PSF
        """
        return self.filenames[self.psf_index]
