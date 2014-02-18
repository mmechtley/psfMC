from __future__ import division
from warnings import warn
import os

import pyfits
import numpy as np
from pymc.StepMethods import AdaptiveMetropolis, DiscreteMetropolis
from pymc.database.pickle import load

from .models import multicomponent_model
from .array_utils import _bad_px_value
from .ModelComponents.PSFSelector import PSFSelector


_default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                      'residual', 'point_source_subtracted')


def model_galaxy_mcmc(obs_file, obsIVM_file, psf_files, psfIVM_files,
                      model_file=None, mag_zeropoint=0,
                      mask_file=None, output_name=None,
                      write_fits=_default_filetypes,
                      chains=1, **kwargs):
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
        model. This should be e.g. a high S/N star
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
                                    db='pickle',
                                    name=db_name)

    for stoch in mc_model.step_method_dict:
        if 'xy' in stoch.__name__:
            mc_model.use_step_method(AdaptiveMetropolis, stoch, interval=10)
        if stoch.__name__ == 'PSF_Index':
            mc_model.use_step_method(DiscreteMetropolis, stoch,
                                     proposal_distribution='Prior')

    db = mc_model.db
    if not os.path.exists(db_name+'.pickle'):
        for chain_num in xrange(chains):
            mc_model.sample(**kwargs)

        # Saves out to pickle file
        db.close()
    else:
        db = load(db_name+'.pickle')
        warn('Database file already exists, skipping sampling')

    # Write model output files
    # TODO: Encode sampler arguments (nchains iter burn etc) in header
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
    header.extend(statscards)
    best_chain, best_samp = header['MPCHAIN'], header['MPSAMP']

    # Record the name of the PSF file used
    psf_selector = [cont for cont in model.containers
                    if isinstance(cont, PSFSelector)].pop()
    model.get_node('PSF_Index').value = \
        db.trace('PSF_Index', best_chain)[best_samp]
    header.set('PSF_IMG', value=psf_selector.value.filename(),
               comment='PSF image of maximum posterior model')

    print 'Saving posterior models'
    # TODO: try/except to handle unknown output types?
    output_data = dict([(ftype, None) for ftype in filetypes])
    if mode in ('maximum', 'MAP'):
        # Set stochastics to their MAP values
        for stoch in model.stochastics - model.observed_stochastics:
            stoch.value = db.trace(stoch.__name__, best_chain)[best_samp]
        for ftype in filetypes:
            output_data[ftype] = np.ma.filled(
                model.get_node(ftype).value, _bad_px_value).copy()
    else:
        total_samples = 0
        for chain in xrange(db.chains):
            chain_samples = db.trace('deviance', chain).length()
            total_samples += chain_samples
            for sample in xrange(chain_samples):
                print 'Processing chain {:d}: {:d}% \r'.format(
                    chain, 100 * sample // chain_samples),
                # Set values of all stochastics
                for stoch in model.stochastics - model.observed_stochastics:
                    stoch.value = db.trace(stoch.__name__, chain)[sample]
                # Accumulate output arrays
                for ftype in filetypes:
                    if output_data[ftype] is None:
                        output_data[ftype] = np.ma.filled(
                            model.get_node(ftype).value, _bad_px_value).copy()
                    else:
                        output_data[ftype] += np.ma.filled(
                            model.get_node(ftype).value, _bad_px_value)
        for ftype in filetypes:
            output_data[ftype] /= total_samples
        print ''

    # Now  save the files
    for ftype in filetypes:
        header.set('OBJECT', value=ftype)
        pyfits.writeto(output_name.format(ftype + '.fits'),
                       output_data[ftype], header=header,
                       clobber=True, output_verify='fix')
    return


def _stats_as_header_cards(db, trace_names=None):
    """
    Collates statistics about the trace database, and returns them in 3-tuple
    key-value-comment format suitable for extending a fits header
    """
    # TODO: better way to make keys. Maybe component.shortname(attr) etc.
    replace_pairs = (('_Sersic', 'SER'), ('_PSF', 'PSF'), ('_Sky', 'SKY'),
                     ('_reff', '_RE'), ('_b', 'B'), ('_index', '_N'),
                     ('_axis_ratio', '_Q'), ('_angle', '_ANG'),
                     ('PSF_Index', 'PSF_IDX'))
    best_chain, best_samp = _max_posterior_sample(db)
    statscards = [('MPCHAIN', best_chain,
                   'Chain index of maximum posterior model'),
                  ('MPSAMP', best_samp,
                   'Sample index of maximum posterior model')]
    for trace_name in sorted(trace_names):
        combined_samps = [db.trace(trace_name, chain)[:]
                          for chain in xrange(db.chains)]
        combined_samps = np.concatenate(combined_samps)
        max_post_val = db.trace(trace_name, best_chain)[best_samp]
        std = np.std(combined_samps, axis=0)
        key = trace_name
        for oldstr, newstr in replace_pairs:
            key = key.replace(oldstr, newstr)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(max_post_val, std)
        except ValueError:
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in max_post_val])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(key, val, 'psfMC model component')]

    # TODO: BPIC might be nice also, but more work to calculate
    # Calculate DIC
    combined_dev = [db.trace('deviance', chain)[:]
                    for chain in xrange(db.chains)]
    combined_dev = np.concatenate(combined_dev)
    mean_dev = np.mean(combined_dev, axis=0)
    dic = 2*mean_dev - db.trace('deviance', best_chain)[best_samp]
    statscards += [('MDL_DIC', dic, 'psfMC Deviance Information Criterion')]

    return statscards


def _max_posterior_sample(db):
    """
    Maximum posterior sample is the sample that minimizes the model deviance
    (i.e. has the highest posterior probability)
    Returns the index of the chain the sample occurs in, and the index of the
    sample within that chain
    """
    min_chain = -1
    min_sample = -1
    min_deviance = 0
    for chain in xrange(db.chains):
        chain_min_sample = np.argmin(db.trace('deviance', chain)[:])
        chain_min_deviance = db.trace('deviance', chain)[chain_min_sample]
        if chain_min_deviance < min_deviance:
            min_deviance = chain_min_deviance
            min_sample = chain_min_sample
            min_chain = chain
    return min_chain, min_sample