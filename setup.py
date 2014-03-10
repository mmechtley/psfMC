from numpy.distutils.core import setup

setup(name='psfMC',
      version='v1.0a1',
      packages=['psfMC', 'psfMC.ModelComponents'],
      requires=['numpy', 'pymc', 'scipy', 'pyfits'])
