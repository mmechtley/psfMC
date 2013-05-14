from psfMC import model_psf_mcmc
import glob, sys

try:
    subfiles = glob.glob(sys.argv[1])
except IndexError:
    subfiles = ['testdata/sci_J0005-0006.fits']
try:
    psffile = sys.argv[2]
except IndexError:
    psffile = 'testdata/sci_psf.fits'

psfIVMfile = psffile.replace('sci', 'ivm')

for subfile in subfiles:
    subIVMfile = subfile.replace('sci', 'ivm')
    model_psf_mcmc(subfile, subIVMfile, psffile, psfIVMfile, magZP=26.2303)