from numpy import outer, modf, clip, where, pi, sin, abs
from .ComponentBase import ComponentBase
from ..array_utils import array_coords


class PSF(ComponentBase):
    """
    Point source component.
    shift_method controls how the point source flux is distributed when sub-
    pixel shifting. Current methods are bilinear and lanczos3
    """
    def __init__(self, xy=None, mag=None, shift_method='lanczos3'):
        self.xy = xy
        self.mag = mag
        self.shift_method = shift_method
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
        if self.shift_method == 'bilinear':
            fracs, ints = modf(clip(self.xy, (0, 0),
                                    (shape[1]-2, shape[0]-2)))
            fluxarr = flux * outer((1-fracs[1], fracs[1]),
                                   (1-fracs[0], fracs[0]))
            arr[ints[1]:ints[1]+2, ints[0]:ints[0]+2] += fluxarr
        elif self.shift_method == 'lanczos3':
            coords = kwargs['coords'] if 'coords' in kwargs \
                else array_coords(arr.shape)
            kern = lanczos(coords-self.xy, 3).prod(axis=1)
            arr += kern.reshape(arr.shape)*flux
        else:
            raise ValueError('Unknown shift method: {}'.format(
                self.shift_method))
        return arr


def sinc(x):
    return where(x != 0, sin(pi*x)/(pi*x), 1.0)


def lanczos(x, a):
    return where(abs(x) < a, sinc(x)*sinc(x/a), 0)