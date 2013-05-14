from psfMC import model_psf_mcmc
import glob, sys
import subprocess
import pymc
import matplotlib.pyplot as pp

try:
    subfiles = glob.glob(sys.argv[1])
except IndexError:
    subfiles = ['testdata/sci_J0005-0006.fits']
try:
    psffile = sys.argv[2]
except IndexError:
    psffile = 'testdata/sci_psf.fits'

psfIVMfile = psffile.replace('sci', 'ivm')


fit_components = [('psf', 60, 70, 60, 70, 18, 23),
                  ('sersic', 60, 70, 60, 70, 22, 27.5,
                   1.5, 8.0, 0.5, 8.0, 0.1, 1.0, 0, 360)]

for subfile in subfiles:
    subIVMfile = subfile.replace('sci', 'ivm')
    dbfile = subfile.replace('sci', 'db').replace('.fits', '')
    model_psf_mcmc(subfile, subIVMfile, psffile, psfIVMfile,
                   fit_components=fit_components,
                   db_name=dbfile, mag_zeropoint=26.2303)

    db = pymc.database.pickle.load(dbfile+'.pickle')
    pp.hist(db.trace('0_psf_mag')[:], bins=20)
    pp.title('PSF mag')
    pp.show()
    pp.hist(db.trace('1_sersic_mag')[:], bins=20)
    pp.title('Sersic mag')
    pp.show()

    runok = subprocess.call(['ds9', subfile.replace('sci','resid'),
                             subfile.replace('sci','model')])

