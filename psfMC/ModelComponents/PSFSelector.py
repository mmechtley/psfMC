from .ComponentBase import ComponentBase
from pymc.distributions import DiscreteUniform
from ..array_utils import preprocess_psf, calculate_psf_variability, pre_fft_psf


class PSFSelector(ComponentBase):
    """
    Selects a PSF from a supplied list. This is not a public-facing class in
    ModelComponents. Rather, psfMC.models.multicomponent_model automatically
    generates a PSFSelector instance for the supplied list of PSFs when setting
    up the model.
    """
    def __init__(self, psflist, ivmlist, data_shape):
        # List-ify psflist and ivmlist if they are single strings
        if isinstance(psflist, basestring) or isinstance(ivmlist, basestring):
            psflist, ivmlist = [psflist], [ivmlist]
        self.filenames = psflist
        self.selected_index = DiscreteUniform('PSF_Index',
                                              lower=0,
                                              upper=len(psflist)-1)
        self.selected_index.fitsname = 'PSF_IDX'

        # Handle PSF bad pixels, normalize
        data_var_pairs = [preprocess_psf(psf, ivm) for psf, ivm
                          in zip(psflist, ivmlist)]
        # Calculate error contribution from mismatch (PSF variability)
        data_var_lists = calculate_psf_variability(*zip(*data_var_pairs))
        # Pre-FFT all psf models to save on per-sample computation
        f_psflist, f_varlist = zip(*[pre_fft_psf(psf, var, data_shape)
                                   for psf, var in zip(*data_var_lists)])
        self.psflist = f_psflist
        self.varlist = f_varlist
        super(PSFSelector, self).__init__()

    def update_trace_names(self, count=None):
        # PSFSelector trace names are set explicitly above in __init__
        return

    def psf(self):
        return self.psflist[self.selected_index]

    def variance(self):
        return self.varlist[self.selected_index]

    def filename(self):
        return self.filenames[self.selected_index]
