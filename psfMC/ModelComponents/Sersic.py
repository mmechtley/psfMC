from __future__ import division
from warnings import warn
from numpy import asarray, exp, cos, sin, deg2rad, sum, log, pi, dot, inf
from scipy.special import gamma, gammaincinv
from .ComponentBase import ComponentBase, StochasticProperty
from ..array_utils import array_coords

try:
    import numexpr as ne
except ImportError:
    warn('numexpr module could not be imported. numexpr is recommended for '
         'optimized (parallel) computation of Sersic profiles.')
    ne = None


class Sersic(ComponentBase):
    """
    Sersic profile component
    """
    _fits_abbrs = [('Sersic', 'SER'), ('reff_b', 'REB'), ('reff', 'RE'),
                   ('index', 'N'), ('angle', 'ANG')]

    xy = StochasticProperty('xy')
    mag = StochasticProperty('mag')
    reff = StochasticProperty('reff')
    reff_b = StochasticProperty('reff_b')
    index = StochasticProperty('index')
    angle = StochasticProperty('angle')

    def __init__(self, xy=None, mag=None, reff=None, reff_b=None,
                 index=None, angle=None, angle_degrees=False):
        super(Sersic, self).__init__()
        self.xy = xy
        self.mag = mag
        self.reff = reff
        self.reff_b = reff_b
        self.index = index
        self.angle = angle
        self.angle_degrees = angle_degrees

    def log_priors(self):
        logp = super(Sersic, self).log_priors()
        # Axis ratio constraint, reff must be bigger than reff_b
        logp += -inf if self.reff_b > self.reff else 0
        return logp

    @staticmethod
    def mag_to_flux(mag, mag_zp):
        """
        Returns total flux of the integrated profile, units relative to mag_zp
        """
        return 10**(-0.4 * (mag - mag_zp))

    @staticmethod
    def kappa(index):
        """
        Sersic profile exponential scaling factor, called either kappa or b_n
        Ciotti & Bertin 1999, A&A, 352, 447 Eqn 5, exact formula!
        """
        return gammaincinv(2 * index, 0.5)

    @staticmethod
    def sb_eff(flux_tot, index, reff, reff_b, kappa=None):
        """
        Returns the surface brightness (in flux units per pixel) at re

        :param flux_tot: Total flux
        :param index: Sersic index n
        :param reff: Effective radius (semi-major axis)
        :param reff_b: Effective radius (semi-minor axis)
        :param mag_zp: (optional) magnitude zeropoint
        :param kappa: (optional) pre-computed normalization constant kappa
        """
        if kappa is None:
            kappa = Sersic.kappa(index)
        return flux_tot / (pi * reff * reff_b * 2*index *
                           exp(kappa + log(kappa) * -2*index) * gamma(2*index))

    def coordinate_sq_radii(self, coords):
        """
        Calculate the generalized ellipse square radii for an array of pixel
        coordinates.

        :param coords: Nx2 array of point coordinates (in rows)
        """
        angle = deg2rad(self.angle) if self.angle_degrees else self.angle
        # Correct for "position angle" CCW of up, instead of right
        angle += 0.5*pi
        sin_ang, cos_ang = sin(angle), cos(angle)

        # Matrix representation of n-D ellipse: en.wikipedia.org/wiki/Ellipsoid
        # inv_xform is inverse scale matrix (1/reff, 0, 0, 1/reff_b)
        # multiplied by inverse rotation matrix (cos, sin, -sin, cos)
        inv_xform = asarray(((cos_ang / self.reff, sin_ang / self.reff),
                             (-sin_ang / self.reff_b, cos_ang / self.reff_b)))
        # TODO: Might be room for optimization here?
        radii = sum(dot(inv_xform, (coords-self.xy).T)**2, axis=0)
        return radii

    def add_to_array(self, arr, mag_zp, **kwargs):
        """
        Add Sersic profile with supplied parameters to a numpy array. Array is
        assumed to be in same units as the zero point, ie the surface brightness
        per pixel is: mag_per_sq_px = -2.5*log10(pixel_value) + mag_zp

        :param arr: Numpy array to add sersic profile to
        :param mag_zp: Magnitude zeropoint (e.g. magnitude of 1 count/second)
        :param kwargs: coords: Optional pre-computed x,y coordinates of each
            element
        """
        coords = kwargs['coords'] if 'coords' in kwargs \
            else array_coords(arr.shape)
        kappa = Sersic.kappa(self.index)
        flux_tot = Sersic.mag_to_flux(self.mag, mag_zp)
        sbeff = self.sb_eff(self.mag, self.index, self.reff, self.reff_b,
                            mag_zp, flux_tot, kappa)

        sq_radii = self.coordinate_sq_radii(coords)
        sq_radii = sq_radii.reshape(arr.shape)

        # Optimization: the square root to get to radii from square radii is
        # combined with the sersic power here
        radius_pow = 0.5 / self.index
        # Optimization: exp(log(a)*b) is faster than a**b or pow(a,b)
        if ne is not None:
            ser_expr = 'sbeff * exp(-kappa * expm1(log(sq_radii)*radius_pow))'
            sb = ne.evaluate(ser_expr)
        else:
            sb = sbeff * exp(-kappa * (exp(log(sq_radii)*radius_pow) - 1))
        # Estimate offset of pixel barycenter from pixel center, in reff units
        # TODO: should delta_r change per-pixel based on ellipse params?
        delta_r = 1 / self.reff
        # Pixel-sized trapezoid having a top with the given normed gradient
        normed_grad = Sersic._normed_grad(sq_radii, radius_pow, kappa)
        bary_offset = delta_r**2 / 12 * normed_grad
        arr += sb * (1 + normed_grad * bary_offset)
        return arr

    @staticmethod
    def _normed_grad(sq_radii, radius_pow, kappa):
        """
        The normalized gradient array (normed grad * surf brightness = grad) for
        a Sersic profile with given square radii array, radius power (0.5/n) and
        Sersic coefficient kappa. The sign is negative, i.e. the formal gradient
        with respect to r as r increases outward.
        """
        # TODO: Use 2D gradient instead of 1D approximation
        # Since square radius is supplied instead of radius, need to be careful
        # about the powers (sqrt happens first so applies to both 1/n and -1)
        if ne is not None:
            grad_expr = '-kappa * 2*radius_pow * ' \
                        'exp(log(sq_radii)*(radius_pow - 0.5))'
            return ne.evaluate(grad_expr)
        else:
            return -kappa * 2*radius_pow * exp(log(sq_radii)*(radius_pow - 0.5))
