from numpy import asarray
from psfMC.ModelComponents import Sky, PSF, Sersic
from psfMC.distributions import Normal, Uniform

zp = 26.2303
totalflux = -5.12 + zp
center = asarray((64, 64))
maxshift = asarray((8, 8))

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Normal(mu=0, tau=100))

# Point source component
PSF(xy=Uniform(lower=center-maxshift, upper=center+maxshift),
    mag=Uniform(lower=totalflux-0.2, upper=totalflux+1.5))

# Sersic profile, modeling the galaxy under the point source
Sersic(xy=Uniform(lower=center-maxshift, upper=center+maxshift),
       mag=Uniform(lower=totalflux, upper=27.5),
       reff=Uniform(lower=2.0, upper=12.0),
       reff_b=Uniform(lower=2.0, upper=12.0),
       index=Uniform(lower=0.5, upper=8),
       angle=Uniform(lower=0, upper=180), angle_degrees=True)

# Second sersic profile, modeling the faint blob to the upper left of the quasar
center = asarray((46, 85.6))
maxshift = asarray((2, 2))
Sersic(xy=Uniform(lower=center-maxshift, upper=center+maxshift),
       mag=Uniform(lower=24.0, upper=25.0),
       reff=Uniform(lower=2.0, upper=4.0),
       reff_b=Uniform(lower=2.0, upper=4.0),
       index=Uniform(lower=0.25, upper=1.5),
       angle=Uniform(lower=92, upper=132), angle_degrees=True)
