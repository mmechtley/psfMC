from __future__ import division, print_function
import numpy as np
from warnings import warn
from emcee.autocorr import AutocorrError


def _between_variance(traces):
    """
    Variance of the individual chain means
    Gelman 2nd edition pg. 303
    :param traces: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = traces.shape

    chain_means = np.mean(traces, axis=0)
    total_mean = np.mean(chain_means)
    scale = nsamples / (nchains - 1)
    return scale * np.sum((chain_means - total_mean)**2)


def _within_variance(traces):
    """
    Mean of the individual chain variances
    Gelman 2nd edition pg. 303
    :param traces: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = traces.shape

    chain_means = np.mean(traces, axis=0)
    col_vars = 1 / (nsamples - 1) * np.sum((traces - chain_means)**2, axis=0)
    return np.mean(col_vars)  # 1 / nchains implicit in mean


def _pooled_posterior_variance(traces):
    """
    Weight average of the within-chain variance and the between-chain variance
    Gelman 2nd edition pg. 303 (Eqn. 11.3)
    :param traces: numpy 2D array, samples in rows, chains in columns
    """
    nsamples, nchains = traces.shape

    return (nsamples - 1) / nsamples * _within_variance(traces) + \
        1 / nsamples * _between_variance(traces)


def potential_scale_reduction(traces):
    """
    So-called R-hat or Potential Scale Reduction Factor (PSRF), the square root
    of the ratio of the marginal posterior variance to the within-chain
    variance.
    Gelman 2nd edition pg. 304
    Brooks & Gelman 1998 eq. 1.1
    :param traces: list of two or more traces (numpy arrays) to analyze
    """
    # TODO: Can be refined by correcting for degrees of freedom (Brooks 1998)
    all_samps = np.column_stack(traces)
    nsamples, nchains = all_samps.shape
    psrf_scale = (nchains + 1) / nchains
    psrf_offset = (1 - nsamples) / (nchains * nsamples)  # negation absorbed
    pooled_var = _pooled_posterior_variance(all_samps)
    within_var = _within_variance(all_samps)
    if within_var == 0:
        return 1.0
    else:
        return np.sqrt(psrf_scale * pooled_var / within_var + psrf_offset)


def num_effective_samples(traces):
    """
    The effective number of samples, ie the number of samples corrected for
    Markov Chain sample autocorrelation. As noted by Gelman, if the number of
    chains is small, this is a crude estimate because the sampling variability
    is high. Always reports min(neff, nsamples * nchains) so as not to claim the
    sampling is more efficient than random.
    Gelman 2nd edition pg. 306 (Eqn. 11.4)
    :param traces: list of two or more traces (numpy arrays) to analyze
    """
    all_samps = np.column_stack(traces)
    nsamples, nchains = all_samps.shape
    pooled_var = _pooled_posterior_variance(all_samps)
    between_var = _between_variance(all_samps)
    # Avoid nan (no between-chain variance) or claim that sampling is better
    # than random (pooled var over-estimate greater than real between-chain)
    if between_var == 0 or pooled_var > between_var:
        return nsamples * nchains
    else:
        return nsamples * nchains * pooled_var / between_var


def check_convergence_psrf(model, chains=None, stochastics=None, psrf_tol=0.05,
                           verbose=0):
    """
    Checks whether chains are converged by calculating the Gelman-Rubin
    Potential Scale Reduction Factor (PSRF) for all traced stochastics
    :param model: model to test
    :param chains: sequence of chain indexes to compare, e.g. [4,5,6,7]
    :param stochastics: List of stochastic (trace) names to consider. If None,
        all traces will be considered.
    :param psrf_tol: Tolerance on PSRF (how close to 1.0 they must be)
    :param verbose: Values >1 indicate to print the psrf for each stochastic
    """
    if chains is None:
        chains = range(model.chains)
    if stochastics is None:
        stochastics = [stoch.__name__ for stoch
                       in model.stochastics - model.observed_stochastics]

    # Automatically assume converged if only one chain is sampled.
    if len(chains) < 2:
        return True

    def is_converged(stoch):
        """
        Test whether a single stochastic is converged (PSRF is within tolerance)
        """
        traces = [model.trace(stoch, chain=cnum)[:] for cnum in chains]
        # Handle multi-dimensional trace variables (like xy center of PSF)
        if len(traces[0].shape) > 1:
            dims = traces[0].shape[1]
            psrf = [potential_scale_reduction([trace[:, dim] for
                                               trace in traces])
                    for dim in range(dims)]
            psrf = np.array(psrf)
        else:
            psrf = potential_scale_reduction(traces)
        if verbose > 0:
            print(stoch, psrf)
        # Converged when psrf is within tolerance
        return np.all(np.abs(psrf - 1) < psrf_tol)

    return all([is_converged(stoch) for stoch in stochastics])


def check_convergence_autocorr(sampler, min_chain_to_tau_ratio=10, verbose=0):
    """
    Use integrated autocorrelation time to estimate whether the chain is
    converged / whether samples are representative
    :param sampler: emcee Sampler object
    :param min_chain_to_tau_ratio: Factor by which the chains must be longer
        than the longest autocorrelation time
    :param verbose: Print parameter autocorrelation values for verbose > 0
    """
    # Use dirty imprecise estimation of autocorrelation time by specifying c=1
    try:
        acorr = sampler.get_autocorr_time(c=1)
    except AutocorrError:
        warn('emcee was unable to estimate the autocorrelation time, assuming '
             'chain is not converged')
        return False
    if verbose > 0:
        print('Autocorrelation times: {}'.format(acorr))

    nsamples = sampler.chain.shape[1]

    return np.all(nsamples > min_chain_to_tau_ratio*acorr)
