import numpy as np
import matplotlib.pyplot as pp
import pyfits
import os
import subprocess
from math import fsum
from scipy.ndimage import shift
from psfMC.ModelComponents import Sersic, PSF
from psfMC.array_utils import array_coords
from timeit import timeit
from string import ascii_uppercase

_sim_feedme = 'sim.feedme'
_sersic_ref_file = 'gfsim_n{:0.1f}.fits.gz'
_psf_ref_shift = np.array((2.2, 2.7))


def _replace_galfit_param(name, value, object=1, fit=True):
    """
    Replaces a parameter value in the galfit configuration file.
    :param name: parameter name, without the following parenthesis
    :param value: new value for the parameter. Best provided as a string
    :param object: For object parameters, which object to change. Galfit
                   numbering, whichs starts with 1. Non-object params (e.g. B)
                   should use default object=1
    :param fit: Whether to fit the parameter (True) or hold fixed (False)
    """
    name, value = str(name), str(value)
    with open(_sim_feedme) as f:
        gf_file = f.readlines()
    # Control parameters only occur once, so 0th index is fine.
    loc = [i for i in range(len(gf_file)) if
           gf_file[i].strip().startswith(name+')')][object-1]
    param_str = gf_file[loc]
    comment = param_str.find('#')
    if name in ascii_uppercase:
        fmt = '{}) {} {}'
        param_str = fmt.format(name, value, param_str[comment:])
    else:
        fmt = '{}) {} {} {}'
        param_str = fmt.format(name, value, '0' if fit else '1',
                               param_str[comment:])
    gf_file[loc] = param_str
    with open(_sim_feedme, 'w') as f:
        f.writelines(gf_file)


def test_sersic(index=4):
    """
    Test un-convolved (raw) Sersic profile against reference (GalFit)
    implementation. Right now, all parameters except index are held fixed.
    """
    sersic_ref_file = _sersic_ref_file.format(index)
    if not os.path.exists(sersic_ref_file):
        nozip_name = sersic_ref_file.replace('.gz', '')
        _replace_galfit_param('B', nozip_name)
        _replace_galfit_param(5, index, object=1, fit=False)
        subprocess.call(['galfit', 'sim.feedme'])
        subprocess.call(['gzip', nozip_name])
    gfmodel = pyfits.getdata(sersic_ref_file)

    gfhdr = pyfits.getheader(sersic_ref_file)
    for key in [key for key in gfhdr if key.startswith('1_')]:
        gfhdr[key] = float(gfhdr[key].split('+/-')[0])
    r_maj = gfhdr['1_RE']
    r_min = r_maj*gfhdr['1_AR']

    mcmodel = np.zeros_like(gfmodel)
    coords = array_coords(mcmodel.shape)
    ser = Sersic(xy=(gfhdr['1_XC']-1, gfhdr['1_YC']-1),
                 mag=gfhdr['1_MAG'], index=gfhdr['1_N'],
                 reff=r_maj, reff_b=r_min,
                 angle=gfhdr['1_PA'], angle_degrees=True)
    ser.add_to_array(mcmodel, mag_zp=gfhdr['MAGZPT'], coords=coords)

    radii = np.sqrt(ser.coordinate_sq_radii(coords))
    radii = radii.reshape(mcmodel.shape)

    print 'Commanded magnitude: {:0.2f} n={:0.1f}'.format(
        gfhdr['1_MAG'], index)
    for model, name in [(gfmodel, 'Galfit'), (mcmodel, ' psfMC')]:
        inside = fsum(model[radii <= 1])
        outside = fsum(model[radii >= 1])
        totalmag = -2.5*np.log10(fsum(model.flat)) + gfhdr['MAGZPT']
        print '{}: Inside: {:0.4f} Outside: {:0.4f} Mag: {:0.2f}'.format(
            name, inside, outside, totalmag)

    abs_error = mcmodel - gfmodel
    frac_error = abs_error / gfmodel

    pp.figure(figsize=(7, 3.5))
    errs = [(abs_error, 'Error'), (frac_error, 'Fractional Error')]
    for step, (err_arr, title) in enumerate(errs):
        pp.subplot((121+step))
        pp.imshow(err_arr, interpolation='nearest', origin='lower')
        pp.colorbar()
        pp.contour(err_arr, levels=[0, ], colors='black')
        pp.contour(frac_error, levels=[-0.01, 0.01], colors='white')
        pp.contour(radii, levels=[1, ], colors='SeaGreen')
        pp.title(title)

    pp.figtext(0.5, 1.0, r'Green: $\Sigma_e$ isophote, ' +
                         'Black: 0% error contour, ' +
                         'White: 1% error contour' +
                         '\nn = {:0.1f}'.format(index),
               va='top', ha='center')

    pp.show()

    def timing_check():
        return ser.add_to_array(mcmodel, mag_zp=gfhdr['MAGZPT'], coords=coords)

    print 'Timing, adding Sersic profile to 128x128 array'
    niter = 1000
    tottime = timeit(timing_check, number=niter)
    print 'Total: {:0.3g}s n={:d} Each: {:0.3g}s'.format(tottime, niter,
                                                         tottime / niter)


def test_psf():
    """
    Test un-convolved PSF model (flux split between 1-4 pixels via sub-pixel
    shifting) vs reference implementation (scipy.ndimage.shift with bilinear
    interpolation)
    """
    print 'Testing PSF component fractional positioning'
    refarr = np.zeros((5, 5))
    # can't put this on the array edge because of boundary modes?
    refarr[1, 1] = 1.0
    # must reverse for scipy.ndimage.shift, since arrays are row, col indexed
    refarr = shift(refarr, _psf_ref_shift[::-1]-1, order=1)

    testarr = np.zeros((5, 5))
    psf = PSF(xy=_psf_ref_shift, mag=0)
    psf.add_to_array(testarr, mag_zp=0)

    assert np.allclose(refarr, testarr)

    mcmodel = np.zeros((128, 128))

    def timing_check():
        return psf.add_to_array(mcmodel, mag_zp=0)

    print 'Timing, adding PSF component to 128x128 array'
    niter = 1000
    tottime = timeit(timing_check, number=niter)
    print 'Total: {:0.3g}s n={:d} Each: {:0.3g}s'.format(tottime, niter,
                                                         tottime / niter)

if __name__ == '__main__':
    test_psf()
    for idx in (0.5, 1.0, 3.1, 4.0, 6.5):
        test_sersic(index=idx)