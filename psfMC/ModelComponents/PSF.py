from numpy import outer, modf, clip
from scipy.special import exp10
from .ComponentBase import ComponentBase


class PSF(ComponentBase):
    """
    Point source component.
    """
    def __init__(self, xy=None, mag=None):
        self.xy = xy
        self.mag = mag
        super(PSF, self).__init__()

    def add_to_array(self, arr, mag_zp, **kwargs):
        """
        Add point source to a numpy array. Array is assumed to be in counts per
        second, ie the brightness of a pixel is
        m = -2.5*log(pixel value) + mag_zp
        Linear interpolation is used for subpixel positions.

        :param arr: Numpy array to add psf to
        :param mag_zp: Magnitude zeropoint (i.e. magnitude of 1 count/second)
        """
        flux = 10**(-0.4 * (self.mag - mag_zp))
        shape = arr.shape
        fracs, ints = modf(clip(self.xy, (0, 0),
                                (shape[1]-2, shape[0]-2)))
        fluxarr = flux * outer((fracs[1], 1-fracs[1]),
                               (fracs[0], 1-fracs[0]))
        # print ints, fluxarr.shape
        arr[ints[1]:ints[1]+2, ints[0]:ints[0]+2] += fluxarr
        return arr
