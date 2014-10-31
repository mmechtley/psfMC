from __future__ import division
from warnings import warn
import os

import pyfits
import numpy as np
import pymc
from pymc.StepMethods import AdaptiveMetropolis, DiscreteMetropolis

from .models import multicomponent_model
from .array_utils import _bad_px_value
from .ModelComponents.PSFSelector import PSFSelector
from .analysis import max_posterior_sample, calculate_dic, chains_are_converged


_default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                      'residual', 'point_source_subtracted')


def model_galaxy_mcmc(obs_file, obsivm_file, psf_files, psfivm_files,
                      model_file=None, mag_zeropoint=0,
                      mask_file=None, output_name=None,
                      write_fits=_default_filetypes,
                      chains=1, backend='pickle',
                      convergence_check=chains_are_converged, max_iterations=1,
                      **kwargs):
    """
    Model the light distribution of a galaxy or galaxies using multi-component
    Markov Chain Monte Carlo parameter estimation.

    :param obs_file: Filename or pyfits HDU containing the observed image, in
        the units specified by the magnitude zeropoint, (usually) electrons per
        second for HST observations).
    :param obsivm_file: Filename or pyfits HDU containing the observed image's
        inverse variance (weight) map. Must already include poisson noise from
        the object, as with multidrizzle ERR weight maps. Consider using
        astroRMS module to include correlated noise in resampled images
    :param psf_files: Filename(s) or pyfits HDU containing the PSF for the
        model. This should be e.g. a high S/N star. If multiple PSF images are
        supplied, the PSF image is treated as a free parameter. Additionally,
        the inter-PSF variance (from breathing or other variability) will be
        calculated propagated into the PSF variance maps.
    :param psfivm_files: Filename(s) or pyfits HDU containing the PSF's inverse
        variance (weight map). Must include poisson noise from the object, such
        as multidrizzle ERR weight maps
    :param model_file: Filename of the model definition file. This should be
        a series of components from psfMC.ModelComponents, with parameters
        supplied as either fixed values or stochastics from psfMC.distributions
    :param mag_zeropoint: Magnitude zeropoint, i.e. the magnitude of one ADU,
        whether in electrons per second (as with published HST zeropoints) or
        whatever funky units the data use.
    :param mask_file: Optional file defining the fitting region. This can be
        used to exclude bad pixels or interloper objects, or confine fitting
        to a smaller region of a large image. Supplied in either fits format
        (where nonzero values indicate exclusion), or ds9 region format.
    :param output_name: Base name for output files (no file extension). By
        default, files are written out containing the raw model, convolved
        model, combined IVM 1/(1/obsIVM + 1/modelIVM), residual of
        observation - model, and the MCMC trace database.
    :param write_fits: List of which fits file types to write. By default, raw
        (unconvolved) model, convolved model, model IVM, and residual.
    :param backend: PyMC database backend to use. pickle is default
    :param convergence_check: Function taking an MCMC model and a list of chain
        indexes as arguments, and returning True or False based on whether the
        model has converged. Default function returns True when all traces have
        potential scale reduction factor within 0.05 of 1.0. Sampling will be
        repeated (increasing effective burn-in period) until convergence check
        is met or until max_iterations iterations are performed
    :param max_iterations: Maximum sampler iterations before convergence is
        enforced. Default is 1, which means sampler halts even if not converged.
    :param kwargs: Further keyword arguments are passed to pyMC.MCMC.sample, and
        can be used to control number of MCMC samples, burn-in period, etc.
        Useful parameters include iter=, burn=, tune_interval=, thin=, etc. See
        pyMC documentation.
    """
    if output_name is None:
        output_name = 'out_' + obs_file.replace('.fits', '').replace('.gz', '')
    output_name += '_{}'

    # TODO: Set these based on total number of unknown components?
    kwargs.setdefault('iter', 6000)
    kwargs.setdefault('burn', 3000)

    # Open database if it exists, otherwise pass backend to create a new one
    db_name = output_name.format('db')
    if os.path.exists(db_name+'.'+backend):
        back_module = getattr(pymc.database, backend)
        db = back_module.load(db_name+'.'+backend)
    else:
        db = backend

    mc_model = multicomponent_model(obs_file, obsivm_file,
                                    psf_files, psfivm_files,
                                    components=model_file,
                                    mag_zp=mag_zeropoint,
                                    mask_file=mask_file,
                                    db=db,
                                    name=db_name)

    # TODO: Add support for resuming. For now, skip sampling if chains exist
    if mc_model.db.chains == 0:
        _set_step_methods(mc_model)

        for samp_iter in xrange(max_iterations):
            # TODO: Is there a way to delete old chains?
            for chain_num in xrange(chains):
                # Seed new values for every independent chain on first iteration
                # On subsequent iterations, load last sample from previous
                if samp_iter == 0:
                    mc_model.draw_from_prior()
                else:
                    mc_model.remember(chain=(samp_iter-1)*chain_num,
                                      trace_index=-1)
                mc_model.sample(**kwargs)

            iter_chains = range(samp_iter*chains, (samp_iter+1)*chains)
            if convergence_check(mc_model, chains=iter_chains):
                break
            else:
                warn('Not yet converged after {:d} iterations ({:d} chains)'
                     .format(samp_iter+1, chains))
        mc_model.db.commit()
    else:
        warn('Database already contains sampled chains, skipping sampling')

    # Write model output files, using only the last `chains' chains.
    post_chains = range(mc_model.db.chains - chains, mc_model.db.chains)
    obs_header = pyfits.getheader(obs_file, ignore_missing_end=True)
    save_posterior_model(mc_model, output_name=output_name,
                         filetypes=write_fits, header=obs_header,
                         chains=post_chains,
                         convergence_check=convergence_check)
    mc_model.db.close()


def save_posterior_model(model, output_name='out_{}', mode='weighted',
                         filetypes=_default_filetypes, header=None,
                         chains=None, convergence_check=chains_are_converged):
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
    :param header: base fits header to include with each image, e.g. a copy of
        the header from the original data
    :param chains: List of chain indexes that sample the final posterior (e.g.
        if initial chains had not converged)
    :param convergence_check: Function taking an MCMC model and a list of chain
        indexes as arguments, and returning True or False based on whether the
        model has converged. Default function returns True when all traces have
        potential scale reduction factor within 0.05 of 1.0.
    """
    if header is None:
        header = pyfits.Header()
    if '{}' not in output_name:
        output_name += '_{}'
    if chains is None:
        chains = range(model.db.chains)

    statscards = _stats_as_header_cards(model, chains=chains,
                                        convergence_check=convergence_check)
    header.extend(statscards, end=True)
    best_chain, best_samp = header['MAPCHAIN'], header['MAPSAMP']

    print 'Saving posterior models'
    # Check to ensure we understand all the requested file types
    determ_names = [node.__name__ for node in model.deterministics]
    unknown_determs = set(filetypes) - set(determ_names)
    if len(unknown_determs) != 0:
        warn('Unknown filetypes requested: {} Output images will not be '
             'generated for these types.'.format(unknown_determs))
        filetypes = set(filetypes) - unknown_determs

    output_data = dict([(ftype, None) for ftype in filetypes])
    if mode in ('maximum', 'MAP'):
        model.remember(chain=best_chain, trace_index=best_samp)
        for ftype in filetypes:
            output_data[ftype] = np.ma.filled(
                model.get_node(ftype).value, _bad_px_value)

    elif mode in ('weighted',):
        total_samples = 0
        for chain in chains:
            chain_samples = model.db.trace('deviance', chain).length()
            total_samples += chain_samples
            for sample in xrange(chain_samples):
                _print_progress(sample, chain, chain_samples)
                model.remember(chain=chain, trace_index=sample)
                # Accumulate output arrays
                for ftype in filetypes:
                    sample_data = np.ma.filled(model.get_node(ftype).value,
                                               _bad_px_value)
                    if output_data[ftype] is None:
                        output_data[ftype] = np.zeros_like(sample_data)
                    if ftype in ('composite_ivm',):
                        sample_data = 1 / sample_data
                    output_data[ftype] += sample_data
        # Take the mean
        for ftype in filetypes:
            output_data[ftype] /= total_samples
            if ftype in ('composite_ivm',):
                output_data[ftype] = 1 / output_data[ftype]

    else:
        warn('Unknown posterior output mode ({}). Posterior model images will '
             'not be saved.'.format(mode))
        return

    # Now  save the files
    for ftype in filetypes:
        header.set('OBJECT', value=ftype)
        pyfits.writeto(output_name.format(ftype + '.fits'),
                       output_data[ftype], header=header,
                       clobber=True, output_verify='fix')
    return


def _stats_as_header_cards(model, chains=None,
                           convergence_check=chains_are_converged):
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
        print 'Processing chain {:d}: {:d}%'.format(
            chain, 100 * sample // chain_samples)
    return


def _set_step_methods(model):
    """
    Set special step methods (for xy positions and discrete variables)
    :param model: multicomponent model
    """
    for stoch in model.step_method_dict:
        if 'xy' in stoch.__name__:
            model.use_step_method(AdaptiveMetropolis, stoch)
        if stoch.__name__ == 'PSF_Index':
            model.use_step_method(DiscreteMetropolis, stoch,
                                  proposal_distribution='Prior')
