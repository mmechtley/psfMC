from psfMC import model_galaxy_mcmc
import glob
import subprocess
import pymc
try:
    import matplotlib.pyplot as pp
except ImportError:
    pp = None


obsfiles = ['testdata/sci_SDSSJ081518.99+103711.5.fits',
            'testdata/sci_J0005-0006.fits']
psffiles = ['testdata/sci_psf16.fits',
            'testdata/sci_psf.fits']

fit_components = [('psf', 60, 70, 60, 70, 15, 23),
                  ('sersic', 60, 70, 60, 70, 18, 27.5,
                   1.5, 8.0, 0.5, 8.0, 0.1, 1.0, 0, 360)]

for obsfile, psffile in zip(obsfiles, psffiles):
    obsIVMfile = obsfile.replace('sci', 'ivm')
    psfIVMfile = psffile.replace('sci', 'ivm')
    output_name = obsfile.replace('sci_', '').replace('.fits', '')
    model_galaxy_mcmc(obsfile, obsIVMfile, psffile, psfIVMfile,
                      fit_components=fit_components,
                      output_name=output_name, mag_zeropoint=26.2303,
                      burn=500, iter=1000)

    db = pymc.database.pickle.load(output_name+'_db.pickle')
    for trace_name in ('0_psf_mag', '1_sersic_mag', '1_sersic_re', '1_sersic_n',
                       '1_sersic_angle', '1_sersic_axisratio'):
        if pp is not None:
            pp.hist(db.trace(trace_name)[:], bins=20)
            pp.title(trace_name)
            pp.show()

    runok = subprocess.call(['ds9', obsfile] + glob.glob(output_name+'_*.fits'))
