from __future__ import division
import numpy as np
from scipy.special import gamma
from .ComponentBase import ComponentBase
from ..array_utils import array_coords

class Sersic(ComponentBase):
    """
    Sersic profile component
    """
    def __init__(self, xy=None, mag=None, reff=None, index=None,
                 axis_ratio=None, angle=None, angle_degrees=True):
        self.xy = xy
        self.mag = mag
        self.reff = reff
        self.index = index
        self.axis_ratio = axis_ratio
        self.angle = angle
        self.angle_degrees = angle_degrees
        super(Sersic, self).__init__()

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
        return flux_tot / (2 * np.pi * self.reff**2 *
                           self.axis_ratio *
                           np.exp(kappa) * self.index *
                           np.power(kappa, -2*self.index) *
                           gamma(2*self.index))

    def kappa(self):
        """
        Sersic profile exponent scaling factor, called either kappa or b_n
        """
        return 1.9992*self.index - 0.3271

    def add_to_array(self, arr, mag_zp, **kwargs):
        """
        Add Sersic profile with supplied parameters to a numpy array. Array is
        assumed to be in counts per second, ie the brightness of a pixel is
        m = -2.5*log(pixel value) + mag_zp

        :param arr: Numpy array to add sersic profile to
        :param mag_zp: Magnitude zeropoint (i.e. magnitude of 1 count/second)
        :param coords: Optional pre-computed x,y coordinates of each element
        """
        coords = kwargs['coords'] if 'coords' in kwargs else array_coords(arr)
        kappa = self.kappa()
        flux_tot = self.total_flux_adu(mag_zp)
        sbeff = self.sb_eff_adu(mag_zp, flux_tot, kappa)
        angle = np.deg2rad(self.angle) if self.angle_degrees else self.angle
        sin_ang, cos_ang = np.sin(angle), np.cos(angle)

        # Matrix representation of n-D ellipse:
        # http://en.wikipedia.org/wiki/Ellipsoid
        M_inv_scale = np.diag((1/self.reff,
                               1/(self.reff*self.axis_ratio)))
        M_rot = np.asarray(((cos_ang, -sin_ang), (sin_ang, cos_ang)))
        # Inverse of a rotation matrix is its transpose
        M_inv_xform = np.dot(M_inv_scale, M_rot.T)

        # TODO: I think there is room for speed improvement here
        # 5.3e-04 seconds for 128x128
        radii = np.sqrt(np.sum(np.dot(M_inv_xform,
                                      (coords-self.xy).T)**2, axis=0))
        radii = radii.reshape(arr.shape)
        # 1.4e-03 seconds for 128x128
        # arr += sbeff * np.exp(-kappa * (np.power(radii, 1/self.index) - 1))
        # 8.8e-04 seconds for 128x128
        arr += sbeff * np.exp(-np.exp(np.log(kappa) +
                                      np.log(radii)*(1/self.index)) - 1)
        return arr
