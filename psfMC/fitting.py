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
from .analysis import max_posterior_sample, calculate_dic


_default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                      'residual', 'point_source_subtracted')


def model_galaxy_mcmc(obs_file, obsIVM_file, psf_files, psfIVM_files,
                      model_file=None, mag_zeropoint=0,
                      mask_file=None, output_name=None,
                      write_fits=_default_filetypes,
                      chains=1, backend='pickle', **kwargs):
    """
    Model the light distribution of a galaxy or galaxies using multi-component
    Markov Chain Monte Carlo parameter estimation.

    :param obs_file: Filename or pyfits HDU containing the observed image, in
        the units specified by the magnitude zeropoint, (usually) electrons per
        second for HST observations).
    :param obsIVM_file: Filename or pyfits HDU containing the observed image's
        inverse variance (weight) map. Must already include poisson noise from
        the object, as with multidrizzle ERR weight maps. Consider using
        astroRMS module to include correlated noise in resampled images
    :param psf_files: Filename(s) or pyfits HDU containing the PSF for the
        model. This should be e.g. a high S/N star. If multiple PSF images are
        supplied, the PSF image is treated as a free parameter. Additionally,
        the inter-PSF variance (from breathing or other variability) will be
        calculated propagated into the PSF variance maps.
    :param psfIVM_files: Filename(s) or pyfits HDU containing the PSF's inverse
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

    db_name = output_name.format('db')
    mc_model = multicomponent_model(obs_file, obsIVM_file,
                                    psf_files, psfIVM_files,
                                    components=model_file,
                                    mag_zp=mag_zeropoint,
                                    mask_file=mask_file,
                                    db=backend,
                                    name=db_name)

    for stoch in mc_model.step_method_dict:
        if 'xy' in stoch.__name__:
            mc_model.use_step_method(AdaptiveMetropolis, stoch, interval=10)
        if stoch.__name__ == 'PSF_Index':
            mc_model.use_step_method(DiscreteMetropolis, stoch,
                                     proposal_distribution='Prior')

    db = mc_model.db
    if not os.path.exists(db_name+'.'+backend):
        for chain_num in xrange(chains):
            mc_model.sample(**kwargs)

        # Saves out to pickle file
        db.close()
    else:
        back_module = getattr(pymc.database, backend)
        db = back_module.load(db_name+'.'+backend)
        db.connect_model(mc_model)
        mc_model.db = db
        warn('Database file already exists, skipping sampling')

    # Write model output files
    obs_header = pyfits.getheader(obs_file, ignore_missing_end=True)
    save_posterior_model(mc_model, db, output_name=output_name,
                         filetypes=write_fits, header=obs_header)


def save_posterior_model(model, db, output_name='out_{}', mode='weighted',
                         filetypes=_default_filetypes, header=None):
    """
    Writes out the posterior model images. Two modes are supported: Maximum a
    posteriori (maximum or MAP) and "weighted average" (weighted). Since
    each MCMC trace represents (ideally) unbiased draws from the posterior
    distribution, the weighted images are simply the (per-pixel) mean of all
    sample images. The "weighting" is handled by the fact that more probable
    locations in parameter space will be more densely populated with samples.
    Sample autocorrelation should be moot, since such correlations are
    distributed roughly equally in parameter space.

    :param model: psfMC model object (MCMC sampler) that defines the model
    :param db: psfMC trace database that defines the (possibly multichain)
               sequence of samples
    :param output_name: base name for output files (no extension)
    :param mode: Either "maximum" or "MAP" (both do MAP), or "weighted".
                 Default is weighted
    :param filetypes: list of filetypes to save out (see model_galaxy_mcmc
                      documentation for a list of possible types
    :param header: base fits header to include with each image, e.g. a copy of
                   the header from the original data
    """
    if header is None:
        header = pyfits.Header()
    if '{}' not in output_name:
        output_name += '_{}'

    stoch_names = [stoch.__name__ for stoch
                   in model.stochastics - model.observed_stochastics]
    statscards = _stats_as_header_cards(db, trace_names=stoch_names)
    header.extend(statscards, end=True)
    best_chain, best_samp = header['MPCHAIN'], header['MPSAMP']

    # Record the name of the PSF file used
    psf_selector = [cont for cont in model.containers
                    if isinstance(cont, PSFSelector)].pop()
    model.get_node('PSF_Index').value = \
        db.trace('PSF_Index', best_chain)[best_samp]
    header.set('PSF_IMG', value=psf_selector.value.filename(),
               comment='PSF image of maximum posterior model')

    print 'Saving posterior models'
    # Check to ensure we understand all the requested file types
    determ_names = [node.__name__ for node in model.deterministics]
    unknown_determs = set(filetypes) - set(determ_names)
    if len(unknown_determs) != 0:
        warn('Unknown filetypes requested: {} '.format(unknown_determs) +
             'Output images will not be generated for these types.')
        filetypes = set(filetypes) - unknown_determs

    output_data = dict([(ftype, None) for ftype in filetypes])
    if mode in ('maximum', 'MAP'):
        model.remember(chain=best_chain, trace_index=best_samp)
        for ftype in filetypes:
            output_data[ftype] = np.ma.filled(
                model.get_node(ftype).value, _bad_px_value).copy()

    elif mode in ('weighted',):
        total_samples = 0
        for chain in xrange(db.chains):
            chain_samples = db.trace('deviance', chain).length()
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
        warn('Unknown posterior output mode ({}). '.format(mode) +
             'Posterior model images will not be saved.')
        return

    # Now  save the files
    for ftype in filetypes:
        header.set('OBJECT', value=ftype)
        pyfits.writeto(output_name.format(ftype + '.fits'),
                       output_data[ftype], header=header,
                       clobber=True, output_verify='fix')
    return


# TODO: better way to make keys. Maybe component.shortname(attr) etc.
# String translation table to change trace names to FITS header keys
_replace_pairs = (('_Sersic', 'SER'), ('_PSF', 'PSF'), ('_Sky', 'SKY'),
                 ('_reff', '_RE'), ('_b', 'B'), ('_index', '_N'),
                 ('_axis_ratio', '_Q'), ('_angle', '_ANG'),
                 ('PSF_Index', 'PSF_IDX'))


def _stats_as_header_cards(db, trace_names=None):
    """
    Collates statistics about the trace database, and returns them in 3-tuple
    key-value-comment format suitable for extending a fits header
    """
    # First get information about the sampler run parameters
    statscards = _section_header('psfMC MCMC SAMPLER PARAMETERS')
    samp_info = db.getstate()['sampler']
    statscards += [
        ('MCITER', samp_info['_iter'], 'number of samples (incl. burn-in)'),
        ('MCBURN', samp_info['_burn'], 'number of burn-in (discarded) samples'),
        ('MCCHAINS', db.chains, 'number of chains run'),
        ('MCTHIN', samp_info['_thin'], 'thin interval (retain every nth)'),
        ('MCTUNE', samp_info['_tune_throughout'],
         'Are AdaptiveMetropolis tuned after burn-in?'),
        ('MCTUNE_N', samp_info['_tune_interval'],
         'AdaptiveMetropolis tuning interval')]

    # Now collect information about the posterior model
    statscards += _section_header('psfMC POSTERIOR MODEL INFORMATION')
    best_chain, best_sample = max_posterior_sample(db)
    statscards += [('MPCHAIN', best_chain,
                    'Chain index of maximum posterior model'),
                   ('MPSAMP', best_sample,
                    'Sample index of maximum posterior model')]
    for trace_name in sorted(trace_names):
        combined_samps = [db.trace(trace_name, chain)[:]
                          for chain in xrange(db.chains)]
        combined_samps = np.concatenate(combined_samps)
        max_post_val = db.trace(trace_name, best_chain)[best_sample]
        std = np.std(combined_samps, axis=0)
        key = trace_name
        for oldstr, newstr in _replace_pairs:
            key = key.replace(oldstr, newstr)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(max_post_val, std)
        except ValueError:
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in max_post_val])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(key, val, 'psfMC model component')]

    dic = calculate_dic(db, best_chain, best_sample)
    statscards += [('MDL_DIC', dic, 'Deviance Information Criterion')]

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
