from psfMC.ModelComponents import Sky, PSF, Sersic
from psfMC.distributions import Normal, Uniform

# We can treat the sky as an unknown component if the subtraction is uncertain
Sky(adu=Normal(mu=0, tau=100))

# Point source component
PSF(xy=Normal(mu=(64, 64), tau=(0.06, 0.06)),
    mag=Uniform(lower=18, upper=24))

# Sersic profile, modeling the galaxy under the point source
Sersic(xy=Normal(mu=(64, 64), tau=(0.06, 0.06)),
       mag=Uniform(lower=22, upper=27.5),
       reff=Uniform(lower=1.0, upper=12.0),
       index=Uniform(lower=0.5, upper=8),
       axis_ratio=Uniform(lower=0.25, upper=1.0),
       angle=Uniform(lower=0, upper=360))

# Second sersic profile, modeling the faint blob to the upper left of the quasar
Sersic(xy=Normal(mu=(46, 86), tau=(0.06, 0.06)),
       mag=Uniform(lower=22, upper=27.5),
       reff=Uniform(lower=1.0, upper=4.0),
       index=Uniform(lower=0.5, upper=8.0),
       axis_ratio=Uniform(lower=0.7, upper=1.0),
       angle=Uniform(lower=0, upper=360))
