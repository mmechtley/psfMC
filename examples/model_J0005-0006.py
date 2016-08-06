"""
Example psfMC model file. The ModelComponents and distributions import
statements are optional (they are injected automatically when the model file is
processed), but their explicit inclusion is recommended.
Model components have several parameters that can generally be supplied as
either a fixed value or as a prior distribution. For instance:
Sersic(..., index=1.0, ...)
will create a Sersic profile with a fixed index of 1 (exponential profile), but:
Sersic(..., index=Uniform(lower=0.5, upper=10.0), ...)
will leave the index as a simulated free parameter with a Uniform prior.
See the docstrings for individual components and distributions for the available
parameters.
"""
from numpy import array
from psfMC.ModelComponents import Configuration, Sky, PointSource, Sersic
from psfMC.distributions import Normal, Uniform, WeibullMinimum

total_mag = 20.83
center = array((64.5, 64.5))
max_shift = array((8, 8))

# The Configuration component is mandatory, and specifies the observation files
Configuration(obs_file='sci_J0005-0006.fits',
              obsivm_file='ivm_J0005-0006.fits',
              psf_files='sci_psf.fits',
              psfivm_files='ivm_psf.fits',
              mask_file='mask_J0005-0006.reg',
              mag_zeropoint=25.9463)

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Normal(loc=0, scale=0.01))

# Point source component
PointSource(xy=Uniform(loc=center - max_shift, scale=2 * max_shift),
            mag=Uniform(loc=total_mag-0.2, scale=0.2+1.5))

# Sersic profile, modeling a galaxy under the point source
Sersic(xy=Uniform(loc=center-max_shift, scale=2*max_shift),
       mag=Uniform(loc=total_mag, scale=27.5-total_mag),
       reff=Uniform(loc=2.0, scale=12.0-2.0),
       reff_b=Uniform(loc=2.0, scale=12.0-2.0),
       index=WeibullMinimum(c=1.5, scale=4),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)

# Second sersic profile, modeling the faint blob to the upper left of the quasar
center = array((46, 85.6))
max_shift = array((5, 5))
Sersic(xy=Uniform(loc=center-max_shift, scale=2*max_shift),
       mag=Uniform(loc=23.5, scale=25.5-23.5),
       reff=Uniform(loc=2.0, scale=8.0-2.0),
       reff_b=Uniform(loc=2.0, scale=8.0-2.0),
       index=WeibullMinimum(c=1.5, scale=4),
       angle=Uniform(loc=0, scale=180), angle_degrees=True)
