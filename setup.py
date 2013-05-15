from numpy.distutils.core import setup

setup(name='psfMC',
      version='0.1',
      packages=['psfMC'],
      requires=['numpy', 'pymc', 'scipy', 'pyfits'])
