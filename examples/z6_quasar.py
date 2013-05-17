from psfMC import model_galaxy_mcmc
import glob
import subprocess
import pymc
try:
    import matplotlib.pyplot as pp
except ImportError:
    pp = None


obsfiles = ['sci_J0005-0006.fits']
psffiles = ['sci_psf.fits']

fit_components = [('sky', -1e-1, 1e-1),
                  ('psf', 60, 70, 60, 70, 18, 23),
                  ('sersic', 60, 70, 60, 70, 20, 27.5,
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
    for trace_name in ('1_psf_mag', '2_sersic_mag', '2_sersic_re', '2_sersic_n',
                       '2_sersic_angle', '2_sersic_axisratio'):
        if pp is not None:
            pp.hist(db.trace(trace_name)[:], bins=20)
            pp.title(trace_name)
            pp.show()

    runok = subprocess.call(['ds9', obsfile] + glob.glob(output_name+'_*.fits'))
