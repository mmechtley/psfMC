from .ComponentBase import ComponentBase
from .PSFSelector import PSFSelector
from ..utils import preprocess_obs, array_coords


class Configuration(ComponentBase):
    """
    Define input images and model control parameters.
    """
    def __init__(self, obs_file, obsivm_file, psf_files, psfivm_files,
                 mask_file=None, mag_zeropoint=0):
        """
        :param obs_file: Filename or FITS HDU containing the observed image,
            in the units specified by the magnitude zeropoint, (usually)
            electrons per second for HST observations).
        :param obsivm_file: Filename or FITS HDU containing the observed
            image's inverse variance (weight) map. Must already include poisson
            noise from the object, as with drizzle ERR weight maps. Consider
            using astroRMS module to estimate correlated noise in resampled
            images
        :param psf_files: Filename(s) or FITS HDU containing the PSF for the
            model. This should be e.g. a high S/N star. If multiple PSF images
            are supplied, the PSF image is treated as a free parameter.
            Additionally, the inter-PSF variance (from breathing or other
            variability) will be calculated propagated into the PSF variance
            maps.
        :param psfivm_files: Filename(s) or FITS HDU containing the PSF's
            inverse variance (weight map). Must include poisson noise from the
            object, such as drizzle ERR weight maps
        :param mask_file: Optional file defining the fitting region. This can be
            used to exclude bad pixels or interloper objects, or confine fitting
            to a smaller region of a large image. Supplied in either fits format
            (where nonzero values indicate exclusion), or ds9 region format.
        :param mag_zeropoint: Magnitude zeropoint, i.e. the magnitude of one
            ADU, whether in electrons per second (as with published HST
            zeropoints) or whatever funky units the data use.
        """
        super(Configuration, self).__init__()
        self.mag_zeropoint = mag_zeropoint

        obs_hdr, obs_data, obs_var, bad_px = \
            preprocess_obs(obs_file, obsivm_file, mask_file)
        self.obs_header = obs_hdr
        self.obs_data = obs_data
        self.obs_var = obs_var
        self.bad_px = bad_px

        # Setup PSF Selector to pick from multiple PSFs
        self.psf_selector = PSFSelector(psf_files, psfivm_files, obs_data.shape)

        # pre-compute data x,y coordinates
        self.coords = array_coords(obs_data.shape)
