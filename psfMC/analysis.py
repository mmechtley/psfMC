from __future__ import division
import numpy as np

def _between_variance(chains):
    """
    Gelman 2nd edition pg. 303
    """
    nsamples, nchains = chains.shape

    phi_cols = np.mean(chains, axis=0)
    phi_phis = np.mean(phi_cols)
    scale = nsamples / (nchains - 1)
    return scale * np.sum((phi_cols - phi_phis)**2)

def _within_variance(chains):
    """
    Gelman 2nd edition pg. 303
    """
    nsamples, nchains = chains.shape

    phi_cols = np.mean(chains, axis=0)
    var_cols = 1 / (nsamples - 1) * np.sum((chains - phi_cols)**2)
    return np.mean(var_cols)

def _marginal_posterior_variance(chains):
    """
    Gelman 2nd edition pg. 303 (Eqn. 11.3)
    """
    nsamples, nchains = chains.shape

    return (nsamples - 1) / nsamples * _within_variance(chains) + \
           1 / nsamples * _between_variance(chains)

def potential_scale_reduction(chains):
    """
    Gelman 2nd edition pg. 304
    """
    all_samps = np.column_stack(chains)
    return np.sqrt(_marginal_posterior_variance(all_samps) /
                   _within_variance(all_samps))

def effective_samples(chains):
    """
    Gelman 2nd edition pg. 306 (Eqn. 11.4)
    """
    all_samps = np.column_stack(chains)
    nsamples, nchains = all_samps.shape
    return nsamples * nchains * _marginal_posterior_variance(all_samps) / \
           _between_variance(all_samps)
