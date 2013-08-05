from __future__ import division
from numpy import asarray, exp, cos, sin, deg2rad, sqrt, sum, log, pi, dot
from scipy.special import gamma
from pymc import Potential
from .ComponentBase import ComponentBase
from ..array_utils import array_coords, debug_timer
import numexpr as ne


class Sersic(ComponentBase):
    """
    Sersic profile component
    """
    def __init__(self, xy=None, mag=None, reff=None, reff_b=None,
                 index=None, angle=None, angle_degrees=False):
        self.xy = xy
        self.mag = mag
        self.reff = reff
        self.reff_b = reff_b
        self.index = index
        self.angle = angle
        self.angle_degrees = angle_degrees

        # Enforce major axis > minor axis.
        # Otherwise rotation angle makes no sense.
        self.axis_ratio_constraint = Potential(logp=Sersic.ab_logp,
                                               name='axis_ratio_constraint',
                                               parents={'major_axis': reff,
                                                        'minor_axis': reff_b},
                                               doc='Axis Ratio Constraint',
                                               verbose=0,
                                               cache_depth=2)

        super(Sersic, self).__init__()

    @staticmethod
    def ab_logp(major_axis, minor_axis):
        # TODO: should be -inf but pymc doesn't like ZeroProbability
        # try some functional form of q?
        return -1e200 if minor_axis > major_axis else 0

    def total_flux_adu(self, mag_zp):
        """
        Returns total flux of the integrated profile, in ADU relative to mag_zp
        """
        return 10**(-0.4 * (self.mag - mag_zp))

    def sb_eff_adu(self, mag_zp, flux_tot=None, kappa=None):
        """
        Returns the surface brightness (in flux units per pixel) at re
        """
        if kappa is None:
            kappa = self.kappa()
        if flux_tot is None:
            flux_tot = self.total_flux_adu(mag_zp)
        return flux_tot / (pi * self.reff * self.reff_b * 2*self.index *
                           exp(kappa + log(kappa) * -2*self.index) *
                           gamma(2*self.index))

    def kappa(self):
        """
        Sersic profile exponential scaling factor, called either kappa or b_n
        """
        # Expansion from Ciotti & Bertin 1999, A&A, 352, 447
        n = self.index
        return (2*n - 1/3 + 4/405*n**-1 + 46/25515*n**-2 + 131/1148175*n**-3
                - 2194697/30690717750*n**-4)

    def coordinate_sq_radii(self, coords):
        """
        Calculate the generalized square radii for an array of pixel
        coordinates.
        """
        angle = deg2rad(self.angle) if self.angle_degrees else self.angle
        # Correct for "position angle" CCW of up, instead of right
        angle += 0.5*pi
        sin_ang, cos_ang = sin(angle), cos(angle)

        # Matrix representation of n-D ellipse: en.wikipedia.org/wiki/Ellipsoid
        # M_inv_xform is inverse scale matrix (1/reff, 0, 0, 1/reff_b)
        # multiplied by inverse rotation matrix (cos, sin, -sin, cos)
        M_inv_xform = asarray(((cos_ang/self.reff, sin_ang/self.reff),
                               (-sin_ang/self.reff_b, cos_ang/self.reff_b)))
        radii = sum(dot(M_inv_xform, (coords-self.xy).T)**2, axis=0)
        return radii

    def add_to_array(self, arr, mag_zp, **kwargs):
        """
        Add Sersic profile with supplied parameters to a numpy array. Array is
        assumed to be in same units as the zero point, ie the surface brightness
        per pixel is:
        m = -2.5*log10(pixel value) + mag_zp

        :param arr: Numpy array to add sersic profile to
        :param mag_zp: Magnitude zeropoint (e.g. magnitude of 1 count/second)
        :param coords: Optional pre-computed x,y coordinates of each element
        """
        coords = kwargs['coords'] if 'coords' in kwargs else array_coords(arr)
        kappa = self.kappa()
        flux_tot = self.total_flux_adu(mag_zp)
        sbeff = self.sb_eff_adu(mag_zp, flux_tot, kappa)

        # TODO: I think there is room for speed improvement here
        # 5.3e-04 seconds for 128x128
        sq_radii = self.coordinate_sq_radii(coords)
        sq_radii = sq_radii.reshape(arr.shape)

        # Optimization: the square root to get to radii from square radii is
        # combined with the sersic power here
        r_power = 0.5/self.index
        # Optimization: exp(log(a)*b) is generally faster than a**b or pow(a,b)
        # 7e-04 seconds for 128x128
        # arr += sbeff * exp(-kappa * (exp(log(sq_radii)*r_power) - 1))
        # 4e-04 seconds for 128x128
        arr += ne.evaluate('sbeff * exp(-kappa * expm1(log(sq_radii)*r_power))')
        return arr
