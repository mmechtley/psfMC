from __future__ import division, print_function
import numpy as np
from warnings import warn
from astropy.io import fits
from ..ModelComponents.PSFSelector import PSFSelector
from . import check_convergence_psrf, max_posterior_sample, calculate_dic

default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                     'residual', 'point_source_subtracted')


def save_posterior_images(model, output_name='out_{}', mode='weighted',
                          filetypes=default_filetypes, bad_px_value=0,
                          chains=None,
                          convergence_check=check_convergence_psrf):
    """
    Writes out the posterior model images. Two modes are supported: Maximum a
    posteriori (maximum or MAP) and "weighted average" (weighted). Since
    each MCMC trace represents (ideally) unbiased draws from the posterior
    distribution, the weighted images are simply the (per-pixel) mean of all
    sample images. The "weighting" is handled by the fact that more probable
    locations in parameter space will be more densely populated with samples.
    Sample autocorrelation should be moot, since such correlations are
    distributed roughly equally in parameter space.

    :param model: psfMC model object (MCMC sampler) that defines the model. May
        have multiple chains
    :param output_name: base name for output files (no extension)
    :param mode: Either "maximum" or "MAP" (both do MAP), or "weighted". Default
        is weighted
    :param filetypes: list of filetypes to save out (see model_galaxy_mcmc
        documentation for a list of possible types
    :param bad_px_value: Value to replace bad pixels with
    :param chains: List of chain indexes that sample the final posterior (e.g.
        if initial chains had not converged)
    :param convergence_check: Function taking an MCMC model and a list of chain
        indexes as arguments, and returning True or False based on whether the
        model has converged. Default function returns True when all traces have
        potential scale reduction factor within 0.05 of 1.0.
    """
    header = model.obs_header
    if '{}' not in output_name:
        output_name += '_{}'
    if chains is None:
        chains = range(model.db.chains)

    statscards = _stats_as_header_cards(model, chains=chains,
                                        convergence_check=convergence_check)
    header.extend(statscards, end=True)
    best_chain, best_samp = header['MAPCHAIN'], header['MAPSAMP']

    print('Saving posterior models')
    # Check to ensure we understand all the requested file types
    determ_names = [node.__name__ for node in model.deterministics]
    unknown_determs = set(filetypes) - set(determ_names)
    if len(unknown_determs) != 0:
        warn('Unknown filetypes requested: {} Output images will not be '
             'generated for these types.'.format(unknown_determs))
        filetypes = set(filetypes) - unknown_determs

    output_data = dict()
    if mode in ('maximum', 'MAP'):
        model.remember(chain=best_chain, trace_index=best_samp)
        for ftype in filetypes:
            output_data[ftype] = np.ma.filled(
                model.get_node(ftype).value, bad_px_value)

    elif mode in ('weighted',):
        total_samples = 0
        for chain in chains:
            chain_samples = model.db.trace('deviance', chain).length()
            total_samples += chain_samples
            for sample in range(chain_samples):
                _print_progress(sample, chain, chain_samples)
                model.remember(chain=chain, trace_index=sample)
                # Accumulate output arrays
                for ftype in filetypes:
                    sample_data = np.ma.filled(model.get_node(ftype).value,
                                               bad_px_value)
                    if ftype not in output_data:
                        output_data[ftype] = np.zeros_like(sample_data)
                    # Accumulate variances rather than IVMs, then invert later
                    if ftype in ('composite_ivm',):
                        sample_data = 1 / sample_data
                    output_data[ftype] += sample_data
        # Take the mean
        for ftype in filetypes:
            output_data[ftype] /= total_samples
            # Invert variance map
            if ftype in ('composite_ivm',):
                output_data[ftype] = 1 / output_data[ftype]

    else:
        warn('Unknown posterior output mode ({}). Posterior model images will '
             'not be saved.'.format(mode))
        return

    # Now save the files
    for ftype in filetypes:
        header.set('OBJECT', value=ftype)
        fits.writeto(output_name.format(ftype + '.fits'),
                     output_data[ftype], header=header,
                     clobber=True, output_verify='fix')
    return


def _stats_as_header_cards(model, chains=None,
                           convergence_check=check_convergence_psrf):
    """
    Collates statistics about the trace database, and returns them in 3-tuple
    key-value-comment format suitable for extending a fits header
    """
    db = model.db
    # First get information about the sampler run parameters
    statscards = _section_header('psfMC MCMC SAMPLER PARAMETERS')
    samp_info = db.getstate()['sampler']
    # Check sampler convergence
    converged = convergence_check(model, chains=chains)
    statscards += [
        ('MCITER', samp_info['_iter'], 'number of samples (incl. burn-in)'),
        ('MCBURN', samp_info['_burn'], 'number of burn-in (discarded) samples'),
        ('MCCHAINS', len(chains), 'number of chains run'),
        ('MCTHIN', samp_info['_thin'], 'thin interval (retain every nth)'),
        ('MCTUNE', samp_info['_tune_throughout'],
         'Are AdaptiveMetropolis tuned after burn-in?'),
        ('MCTUNE_N', samp_info['_tune_interval'],
         'AdaptiveMetropolis tuning interval'),
        ('MCCONVRG', converged, 'Has MCMC sampler converged?')]

    # Now collect information about the posterior model
    statscards += _section_header('psfMC POSTERIOR MODEL INFORMATION')
    best_chain, best_sample = max_posterior_sample(db, chains)
    statscards += [
        ('MAPCHAIN', best_chain, 'Chain index of maximum posterior model'),
        ('MAPSAMP', best_sample, 'Sample index of maximum posterior model')]

    stochastics = model.stochastics - model.observed_stochastics
    for stoch in sorted(stochastics, key=lambda s: s.__name__):
        trace_name = stoch.__name__
        fits_key = stoch.fitsname
        combined_samps = [db.trace(trace_name, chain)[:] for chain in chains]
        combined_samps = np.concatenate(combined_samps)
        max_post_val = db.trace(trace_name, best_chain)[best_sample]
        std = np.std(combined_samps, axis=0)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(max_post_val, std)
        except ValueError:
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in max_post_val])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(fits_key, val, 'psfMC model component')]

    dic = calculate_dic(db, chains, best_sample, best_chain)
    statscards += [('MDL_DIC', dic, 'Deviance Information Criterion')]

    # Record the name of the PSF file used in the MP model
    psf_selector = [cont for cont in model.containers
                    if isinstance(cont, PSFSelector)].pop()
    model.remember(chain=best_chain, trace_index=best_sample)
    statscards += [('PSF_IMG', psf_selector.value.filename(),
                    'PSF image of maximum posterior model')]

    return statscards


def _section_header(section_name):
    """
    Blank fits header cards for a section header. As in drizzle, one blank line,
    a line with the section name as a comment, then one more blank.
    """
    return [('', '', ''), ('', '/ ' + section_name, ''), ('', '', '')]


def _print_progress(sample, chain, chain_samples):
    if sample % (chain_samples // 100) == 0:
        print('Processing chain {:d}: {:d}%'.format(
            chain, 100 * sample // chain_samples))
    return