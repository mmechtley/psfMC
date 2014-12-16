from psfMC import model_galaxy_mcmc
import glob
import subprocess
import pymc
import pyfits
try:
    import matplotlib.pyplot as pp
except ImportError:
    pp = None

# Arguments that will be passed to the MCMC sampler. iter and burn are selected
# to allow this demonstration to run quickly. They are almost certainly too
# small to provide proper convergence.
mc_args = {'burn': 5000, 'iter': 10000, 'chains': 4,
           'tune_interval': 250, 'max_iterations': 1}

# These lists could be longer, to define more quasar+PSF pairs to run
model_files = ['model_J0005-0006.py']

for model_file in model_files:
    # Do some string manipulation to infer filenames for additional files
    output_name = model_file.replace('model', 'out').replace('.py', '')

    # Run the MCMC sampler
    model_galaxy_mcmc(model_file, output_name=output_name, **mc_args)

    # Once sampling has completed, display some example plots
    db = pymc.database.pickle.load(output_name+'_db.pickle')
    for trace_name in ('0_Sky_adu', '1_PSF_mag', '2_Sersic_mag',
                       '2_Sersic_reff', '2_Sersic_index', '2_Sersic_angle'):
        if pp is not None:
            pp.hist(db.trace(trace_name)[:], bins=20)
            pp.title(trace_name)
            pp.show()
    db.close()

    # Also try to open a ds9 window with all the input and output fits images
    in_images = glob.glob('sci_*.fits')
    out_images = glob.glob(output_name+'*.fits')
    runok = subprocess.call(['ds9'] + in_images + out_images)
