from numpy.distutils.core import setup

setup(name='psfMC',
      version='0.1',
      packages=['psfMC', 'psfMC.ModelComponents'],
      requires=['numpy', 'pymc', 'scipy', 'pyfits'])
