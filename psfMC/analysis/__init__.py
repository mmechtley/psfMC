"""
This module contains functionality for analyzing posterior distributions from
sample databases, including generating posterior model images, calculating
statistics and convergence diagnostics, and plotting
"""
from .statistics import check_convergence_autocorr
from .plotting import plot_hist, plot_trace, plot_autocorr, corner_plot
from .images import save_posterior_images
