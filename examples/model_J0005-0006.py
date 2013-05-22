from psfMC.ModelComponents import Sky, PSF, Sersic
from psfMC.distributions import Normal, Uniform

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Normal(mu=0, tau=100))

# Point source component
PSF(xy=Uniform(lower=(60, 60), upper=(70, 70)),
    mag=Uniform(lower=18, upper=24))

# Sersic profile, modeling the galaxy under the point source
Sersic(xy=Uniform(lower=(60, 60), upper=(70, 70)),
       mag=Uniform(lower=22, upper=27.5),
       reff=Uniform(lower=1.0, upper=12.0),
       index=Uniform(lower=0.5, upper=8),
       axis_ratio=Uniform(lower=0.25, upper=1.0),
       angle=Uniform(lower=0, upper=360))

# Second sersic profile, modeling the faint blob to the upper left of the quasar
Sersic(xy=Uniform(lower=(42, 82), upper=(50, 90)),
       mag=Uniform(lower=22, upper=27.5),
       reff=Uniform(lower=1.0, upper=4.0),
       index=Uniform(lower=0.5, upper=8.0),
       axis_ratio=Uniform(lower=0.7, upper=1.0),
       angle=Uniform(lower=0, upper=360))
