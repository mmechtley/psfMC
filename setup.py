import os
import re
from numpy.distutils.core import setup

_packages = ['psfMC', 'psfMC.ModelComponents', 'psfMC.analysis']
_scripts = ['scripts/plot_chain', 'scripts/corner_plot']
_requires = ['numpy', 'scipy', 'astropy', 'emcee', 'matplotlib', 'corner',
             'six']


def _version_from_init():
    main_init = os.path.join(_packages[0], '__init__.py')
    ver_pat = re.compile('__version__\s*=\s*\'([^\']*)\'')
    with open(main_init) as initfile:
        for line in initfile.readlines():
            match = ver_pat.match(line)
            if match is not None:
                return match.group(1)
    return ''

_version = _version_from_init()

setup(name=_packages[0],
      version=_version,
      author='Matt Mechtley',
      author_email='matt@cherem.org',
      url='https://github.com/mmechtley/psfMC',
      license='BSD 2-clause',
      description='Markov Chain Monte Carlo modeling of 2-dimensional surface '
                  'brightness distributions of quasar host galaxies',
      packages=_packages,
      scripts=_scripts,
      requires=_requires)
