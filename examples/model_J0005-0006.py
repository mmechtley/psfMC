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
from psfMC.ModelComponents import Configuration, Sky, PSF, Sersic
from psfMC.distributions import Normal, Uniform, Weibull

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
# pyMC Normal distributions are parameterized by mu, tau (= 1/sigma**2)
Sky(adu=Normal(mu=0, tau=100))

# Point source component
PSF(xy=Uniform(lower=center-max_shift, upper=center+max_shift),
    mag=Uniform(lower=total_mag-0.2, upper=total_mag+1.5))

# Sersic profile, modeling a galaxy under the point source
Sersic(xy=Uniform(lower=center-max_shift, upper=center+max_shift),
       mag=Uniform(lower=total_mag, upper=27.5),
       reff=Uniform(lower=2.0, upper=12.0),
       reff_b=Uniform(lower=2.0, upper=12.0),
       index=Weibull(alpha=1.5, beta=4),
       angle=Uniform(lower=0, upper=180), angle_degrees=True)

# Second sersic profile, modeling the faint blob to the upper left of the quasar
center = array((46, 85.6))
max_shift = array((5, 5))
Sersic(xy=Uniform(lower=center-max_shift, upper=center+max_shift),
       mag=Uniform(lower=23.5, upper=25.5),
       reff=Uniform(lower=2.0, upper=8.0),
       reff_b=Uniform(lower=2.0, upper=8.0),
       index=Weibull(alpha=1.5, beta=4),
       angle=Uniform(lower=0, upper=180), angle_degrees=True)
