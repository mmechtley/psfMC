from psfMC import model_galaxy_mcmc, load_database
from psfMC.analysis import plot_hist
import glob
import subprocess

# Arguments that will be passed to the MCMC sampler. iter and burn are selected
# to allow this demonstration to run quickly. They are almost certainly too
# small to provide proper convergence.
mc_args = {'burn': 50, 'iterations': 100, 'chains': 250}

# This list could be longer, to define more model files to run
model_files = ['model_J0005-0006.py']

for model_file in model_files:
    # Do some string manipulation create the base output filename
    output_name = model_file.replace('model', 'out').replace('.py', '')

    # Run the MCMC sampler
    model_galaxy_mcmc(model_file, output_name=output_name, **mc_args)

    # Once sampling has completed, display some example plots
    db_file = output_name + '_db.fits'
    db = load_database(db_file)
    # for trace_name in ('0_Sky_adu', '1_PSF_mag', '1_PSF_2_Sersic_magdiff'):
    #     plot_hist(trace_name, db_file, model=model_file,
    #               chains=range(-mc_args['chains'], 0))

    # Also try to open a ds9 window with all the input and output fits images
    in_images = glob.glob('sci_*.fits')
    out_images = glob.glob(output_name+'*.fits')
    runok = subprocess.call(['ds9'] + in_images + out_images)
