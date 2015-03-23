__version__ = 'v1.0b4'

from .fitting import model_galaxy_mcmc
from .models import multicomponent_model
from .analysis import save_posterior_images, potential_scale_reduction, \
    num_effective_samples
