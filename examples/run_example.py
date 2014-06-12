from psfMC import model_galaxy_mcmc
import glob
import subprocess
import pymc
import pyfits
try:
    import matplotlib.pyplot as pp
except ImportError:
    pp = None

# Magnitude zeropoints for two HST filters
mag_zps = {'F125W': 26.2303, 'F160W': 25.9463}
# Arguments that will be passed to the MCMC sampler. iter and burn are selected
# to allow this demonstration to run quickly. They are almost certainly too
# small to provide proper convergence.
mc_args = {'burn': 5000, 'iter': 10000, 'chains': 4,
           'tune_interval': 250, 'max_iterations': 1}

# These lists could be longer, to define more quasar+PSF pairs to run
obsfiles = ['sci_J0005-0006.fits']
psffiles = ['sci_psf.fits']

for obsfile, psffile in zip(obsfiles, psffiles):
    # Do some string manipulation to infer filenames for additional files
    obsivm_file = obsfile.replace('sci', 'ivm')
    psfivm_file = psffile.replace('sci', 'ivm')
    mask_file = obsfile.replace('sci', 'mask').replace('.fits', '.reg')
    model_file = obsfile.replace('sci', 'model').replace('.fits', '.py')
    output_name = obsfile.replace('sci', 'out').replace('.fits', '')

    filt = pyfits.getval(obsfile, 'FILTER')
    mag_zp = mag_zps[filt]

    # Run the MCMC sampler
    model_galaxy_mcmc(obsfile, obsivm_file, psffile, psfivm_file,
                      model_file=model_file, mask_file=mask_file,
                      output_name=output_name, mag_zeropoint=mag_zp,
                      **mc_args)

    # Once sampling has completed, display some example plots
    db = pymc.database.pickle.load(output_name+'_db.pickle')
    for trace_name in ('0_Sky_adu', '1_PSF_mag', '2_Sersic_mag',
                       '2_Sersic_reff', '2_Sersic_index', '2_Sersic_angle'):
        if pp is not None:
            pp.hist(db.trace(trace_name)[:], bins=20)
            pp.title(trace_name)
            pp.show()

    # Also try to open a ds9 window with all the input and output fits images
    out_images = glob.glob(output_name+'_*.fits')
    runok = subprocess.call(['ds9', obsfile, obsivm_file] + out_images)
