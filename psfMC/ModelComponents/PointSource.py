from numpy import array, abs, pi, sin, prod, clip, where, round
from .ComponentBase import ComponentBase, StochasticProperty
from ..array_utils import array_coords


class PointSource(ComponentBase):
    """
    Point source component
    shift_method controls how the point source flux is distributed when sub-
    pixel shifting. Current methods are bilinear and lanczos3.
    xy position is 0-based (like numpy array) not 1-based (like FITS)
    """
    _fits_abbrs = [('PointSource', 'PS')]

    xy = StochasticProperty('xy')
    mag = StochasticProperty('mag')

    def __init__(self, xy=None, mag=None, shift_method='lanczos3'):
        super(PointSource, self).__init__()
        self.xy = xy
        self.mag = mag
        self.shift_method = shift_method

    def add_to_array(self, arr, mag_zp, **kwargs):
        """
        Add point source to a numpy array. Array values are assumed to be in
        units such that the brightness of a pixel is
        m = -2.5*log(pixel value) + mag_zp

        :param arr: Numpy array to add psf to
        :param mag_zp: Magnitude zeropoint (i.e. magnitude of 1 count/second)
        :param kwargs: coords: pre-computed array of pixel coordinates
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
    # TODO: log_priors should just return -inf instead of using clip
    clipped_pos = clip(position[::-1],
                       kern_radius - 0.5,
                       array_shape - (kern_radius + 0.5))
    min_pos = round(clipped_pos - kern_radius).astype(int)
    max_pos = round(clipped_pos + kern_radius).astype(int)
    return slice(min_pos[0], max_pos[0]+1), slice(min_pos[1], max_pos[1]+1)


def sinc(x):
    """
    sinc function
    """
    return where(x != 0, sin(pi*x)/(pi*x), 1.0)


def lanczos(x, a):
    """
    1D Lanczos kernel
    :param x: positions to evaluate
    :param a: size of lanczos window
    """
    return where(abs(x) < a, sinc(x)*sinc(x/a), 0)
