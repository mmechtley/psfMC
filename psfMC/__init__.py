__version__ = 'v1.0b1'

from .fitting import model_galaxy_mcmc, save_posterior_model
from .models import multicomponent_model
from .analysis import potential_scale_reduction, num_effective_samples
