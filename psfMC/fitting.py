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
                      **kwargs):
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
        output_name = obs_file.replace('.fits', '')
    output_name += '_{}'

    # TODO: Set these based on total number of unknown components
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
        mc_model.sample(**kwargs)

        ## Saves out to pickle file
        db.close()
    else:
        db = load(db_name+'.pickle')
        warn('Database file already exists, skipping sampling')

    # Write mean model output files
    obsHeader = pyfits.getheader(obs_file, ignore_missing_end=True)
    write_ml_model(mc_model, db, basename=output_name,
                   filetypes=write_fits, header=obsHeader)


def write_ml_model(model, db, basename='mcmc', filetypes=_default_filetypes,
                   samples_slice=slice(0, -1), header=None):
    if header is None:
        header = pyfits.Header()
    if '{}' not in basename:
        basename += '_{}'

    stoch_names = [stoch.__name__ for stoch
                   in model.stochastics - model.observed_stochastics]
    statscards = _stats_as_header_cards(db, trace_names=stoch_names,
                                        trace_slice=samples_slice)
    header.extend(statscards)

    # Set model stochastic values to their trace means
    for stoch in model.stochastics - model.observed_stochastics:
        trace = db.trace(stoch.__name__)[samples_slice]
        #stoch.value = _max_likelihood_value(trace)
        best_index = np.argmin(db.trace('deviance')[:])
        stoch.value = db.trace(stoch.__name__)[best_index]

    # Find name of PSF file used
    psf_selector = [cont for cont in model.containers
                    if isinstance(cont, PSFSelector)].pop()
    header.set('PSF_IMG', value=psf_selector.value.filename(),
               comment='psfMC maximum likelihood PSF image')

    # TODO: BPIC might be better, but more work to calculate
    # Calculate DIC
    mean_dev = np.mean(db.trace('deviance')[samples_slice], axis=0)
    dic = 2*mean_dev - model.deviance
    header.set('MDL_DIC', value=dic,
               comment='psfMC Deviance Information Criterion')

    # Save out requested file types
    for out_type in filetypes:
        try:
            outputData = np.ma.filled(model.get_node(out_type).value,
                                      _bad_px_value)
        except AttributeError:
            warn(('Unable to find model parameter named {}. No output will ' +
                  'be written for file type.').format(out_type))
            continue

        header.set('OBJECT', value=out_type)
        pyfits.writeto(basename.format(out_type + '.fits'),
                       outputData.copy(), header=header,
                       clobber=True, output_verify='fix')


def _stats_as_header_cards(db, trace_names=None, trace_slice=slice(0, -1)):
    # TODO: better way to make keys. Maybe component.shortname(attr) etc.
    replace_pairs = (('_Sersic', 'SER'), ('_PSF', 'PSF'), ('_Sky', 'SKY'),
                     ('_reff', '_RE'), ('_b', 'B'), ('_index', '_N'),
                     ('_axis_ratio', '_Q'), ('_angle', '_ANG'),
                     ('PSF_Index', 'PSF_IDX'))
    statscards = []
    for trace_name in sorted(trace_names):
        trace = db.trace(trace_name)[trace_slice]
        max_likely = _max_likelihood_value(trace)
        std = np.std(trace, axis=0)
        key = trace_name
        for oldstr, newstr in replace_pairs:
            key = key.replace(oldstr, newstr)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(max_likely, std)
        except ValueError:
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in max_likely])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(key, val, 'psfMC model component')]
    return statscards


def _max_likelihood_value(trace):
    # Discrete-valued stochastics should be set to most common value
    if trace.dtype.kind in 'iu':
        return trace[np.bincount(trace).argmax(axis=0)]
    else:
        return np.mean(trace, axis=0)
