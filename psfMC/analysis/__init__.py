"""
This module contains functionality for analyzing posterior distributions from
sample databases, including generating posterior model images, calculating
statistics and convergence diagnostics, and plotting
"""
from .statistics import potential_scale_reduction, num_effective_samples, \
    max_posterior_sample, calculate_dic, check_convergence_psrf, chain_autocorr
from .plotting import plot_hist, plot_trace, corner_plot
from .images import save_posterior_images
