import numpy as np
import matplotlib.pyplot as pp
import pyfits
from psfMC.ModelComponents import Sersic


if __name__ == '__main__':
    gfmodel = pyfits.getdata('gfsim.fits')
    gfhdr = pyfits.getheader('gfsim.fits')
    for key in [key for key in gfhdr if key.startswith('1_')]:
        gfhdr[key] = float(gfhdr[key].split('+/-')[0])
    mcmodel = np.zeros_like(gfmodel)
    ser = Sersic(xy=(gfhdr['1_XC']-1, gfhdr['1_YC']-1),
                 mag=gfhdr['1_MAG'], index=gfhdr['1_N'],
                 reff=gfhdr['1_RE'], reff_b=gfhdr['1_RE']*gfhdr['1_AR'],
                 angle=gfhdr['1_PA']+90, angle_degrees=True)
    ser.add_to_array(mcmodel, mag_zp=gfhdr['MAGZPT'])

    pp.imshow((mcmodel - gfmodel)/gfmodel, interpolation='nearest')
    pp.colorbar()
    pp.title('Fractional Error')
    pp.show()