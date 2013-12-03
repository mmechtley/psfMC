from __future__ import division
import numpy as np


def _between_variance(chains):
    """
    Variance of the individual chain means
    Gelman 2nd edition pg. 303
    :param chains: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = chains.shape

    chain_means = np.mean(chains, axis=0)
    total_mean = np.mean(chain_means)
    scale = nsamples / (nchains - 1)
    return scale * np.sum((chain_means - total_mean)**2)


def _within_variance(chains):
    """
    Mean of the individual chain variances
    Gelman 2nd edition pg. 303
    :param chains: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = chains.shape

    chain_means = np.mean(chains, axis=0)
    col_vars = 1 / (nsamples - 1) * np.sum((chains - chain_means)**2, axis=0)
    return np.mean(col_vars)


def _marginal_posterior_variance(chains):
    """
    Weight average of the within-chain variance and the between-chain variance
    Gelman 2nd edition pg. 303 (Eqn. 11.3)
    :param chains: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = chains.shape

    return (nsamples - 1) / nsamples * _within_variance(chains) + \
        1 / nsamples * _between_variance(chains)


def potential_scale_reduction(chains):
    """
    So-called R-hat, the square root of the ratio of the marginal posterior
    variance to the within-chain variance.
    Gelman 2nd edition pg. 304
    :param chains: list of two or more traces (numpy arrays) to analyze
    """
    all_samps = np.column_stack(chains)
    return np.sqrt(_marginal_posterior_variance(all_samps) /
                   _within_variance(all_samps))


def num_effective_samples(chains):
    """
    The effective number of samples, ie the number of samples corrected for
    Markov Chain sample autocorrelation. As noted by Gelman, if the number of
    chains is small, this is a crude estimate because the sampling variability
    is high.
    Usually reported as min(n_eff, nsamps*nchains).
    Gelman 2nd edition pg. 306 (Eqn. 11.4)
    :param chains: list of two or more traces (numpy arrays) to analyze
    """
    all_samps = np.column_stack(chains)
    nsamples, nchains = all_samps.shape
    return nsamples * nchains * _marginal_posterior_variance(all_samps) / \
        _between_variance(all_samps)
