from numpy import array, abs, pi, sin, prod, clip, where, round
from .ComponentBase import ComponentBase
from ..array_utils import array_coords


class PSF(ComponentBase):
    """
    Point source component.
    shift_method controls how the point source flux is distributed when sub-
    pixel shifting. Current methods are bilinear and lanczos3.
    xy position is 0-based (like numpy array) not 1-based (like FITS)
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
        coords = kwargs['coords'] if 'coords' in kwargs \
            else array_coords(arr.shape)
        # Explicitly taking a view, then setting shape raises an error if the
        # result would involve copying the array (which we want to avoid)
        coords_2d = coords.view()
        coords_2d.shape = arr.shape + (2,)

        if self.shift_method == 'bilinear':
            kern_slice = minimal_slice(self.xy, 0.5, arr.shape)
            kern_coords = coords_2d[kern_slice]
            diffs = kern_coords - self.xy
            kern = prod(1 - abs(diffs), axis=-1)
        elif self.shift_method == 'lanczos3':
            kern_slice = minimal_slice(self.xy, 3, arr.shape)
            kern_coords = coords_2d[kern_slice]
            diffs = kern_coords - self.xy
            kern = prod(lanczos(diffs, 3), axis=-1)
        else:
            raise ValueError('Unknown shift method: {}'
                             .format(self.shift_method))

        flux = 10**(-0.4 * (self.mag - mag_zp))
        arr[kern_slice] += kern * flux
        return arr


def minimal_slice(position, kern_radius, array_shape):
    """
    Get a slice representing the minimum array section that includes the given
    position, padded by the given kern_radius. That is, exactly those pixels
    whose center coordinates are within +/- kern_radius of position.
    :param position: Position in xy coordinate order
    :param kern_radius: Radius of the kernel (symmetric about position)
    :param array_shape: Shape of embedding numpy array (yx order)
    :return: 2D slice tuple in yx order
    """
    kern_radius = array(kern_radius)
    array_shape = array(array_shape)
    # First, clip the position to stay more than kern_radius from the edge.
    # position is in xy order, but array shape and returned slice are yx.
    clipped_pos = clip(position[::-1],
                       kern_radius - 0.5,
                       array_shape - (kern_radius + 0.5))
    min_pos = round(clipped_pos - kern_radius)
    max_pos = round(clipped_pos + kern_radius)
    return slice(min_pos[0], max_pos[0]+1), slice(min_pos[1], max_pos[1]+1)


def sinc(x):
    return where(x != 0, sin(pi*x)/(pi*x), 1.0)


def lanczos(x, a):
    return where(abs(x) < a, sinc(x)*sinc(x/a), 0)