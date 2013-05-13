from psfMC import model_psf_mcmc
import glob, sys

subfiles = glob.glob(sys.argv[1])

psffile = sys.argv[2]
psfIVMfile = psffile.replace('sci', 'ivm')

for subfile in subfiles:
    subIVMfile = subfile.replace('sci', 'ivm')
    model_psf_mcmc(subfile, subIVMfile, psffile, psfIVMfile)