from psfMC import model_galaxy_mcmc
import glob
import subprocess
import pymc
import pyfits
try:
    import matplotlib.pyplot as pp
except ImportError:
    pp = None

magzp = {'F125W':26.2303, 'F160W':25.9463}

obsfiles = ['sci_J0005-0006.fits']
psffiles = ['sci_psf.fits']

for obsfile, psffile in zip(obsfiles, psffiles):
    obsIVMfile = obsfile.replace('sci', 'ivm')
    psfIVMfile = psffile.replace('sci', 'ivm')
    maskfile = obsfile.replace('sci', 'mask').replace('.fits','.reg')
    model_file = obsfile.replace('sci', 'model').replace('.fits', '.py')
    output_name = obsfile.replace('sci', 'out').replace('.fits', '')

    filter = pyfits.getval(obsfile, 'FILTER')
    model_galaxy_mcmc(obsfile, obsIVMfile, psffile, psfIVMfile,
                      model_file=model_file, mask_file=maskfile,
                      output_name=output_name, mag_zeropoint=magzp[filter],
                      burn=5000, iter=10000, chains=4)

    db = pymc.database.pickle.load(output_name+'_db.pickle')
    for trace_name in ('0_Sky_adu', '1_PSF_mag', '2_Sersic_mag',
                       '2_Sersic_reff', '2_Sersic_index', '2_Sersic_angle'):
        if pp is not None:
            pp.hist(db.trace(trace_name)[:], bins=20)
            pp.title(trace_name)
            pp.show()

    runok = subprocess.call(['ds9', obsfile, obsIVMfile] +
                            glob.glob(output_name+'_*.fits'))
