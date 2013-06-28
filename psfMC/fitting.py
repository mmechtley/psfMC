from __future__ import division
from warnings import warn

import pyfits
import numpy as np
from pymc.MCMC import MCMC

from .models import multicomponent_model
from .array_utils import mask_bad_pixels

try:
    import pyregion
except ImportError:
    pyregion = None


def model_galaxy_mcmc(obs_file, obsIVM_file, psf_file, psfIVM_file,
                      model_file=None, mag_zeropoint=0,
                      mask_file=None, output_name=None,
                      write_fits=('raw_model', 'convolved_model',
                                  'composite_ivm', 'residual'),
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
    :param psf_file: Filename or pyfits HDU containing the PSF for the model.
        This should be e.g. a high S/N star
    :param psfIVM_file: Filename or pyfits HDU containing the PSF's inverse
        variance (weight map). Must include poisson noise from the object, such
        as multidrizzle ERR weight maps
    :param model_file: Filename of the model definition file. This should be
        a series of components from psfMC.ModelComponents, with parameters
        supplied as either fixed values or stochastics from psfMC.distributions
    :param mag_zeropoint: Magnitude zeropoint, i.e. the magnitude of one ADU,
        whether in electrons per second (as with published HST zeropoints) or
        whatever funky units the data use.
    :param mask_file: Optional ds9 region file defining the fitting region.
        This can be used to exclude bad pixels or interloper objects, or
        confine fitting to a smaller region of a large image.
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

    obsData = pyfits.getdata(obs_file, ignore_missing_end=True)
    obsDataIVM = pyfits.getdata(obsIVM_file, ignore_missing_end=True)
    psfData = pyfits.getdata(psf_file, ignore_missing_end=True)
    psfDataIVM = pyfits.getdata(psfIVM_file, ignore_missing_end=True)

    obsData, obsDataIVM, psfData, psfDataIVM = mask_bad_pixels(
        obsData, obsDataIVM, psfData, psfDataIVM)

    # FIXME: Masks should use masked array, but this breaks fitting?
    if mask_file is not None:
        if pyregion is not None:
            hdr = pyfits.getheader(obs_file)
            regfilt = pyregion.open(mask_file).as_imagecoord(hdr).get_filter()
            mask = regfilt.mask(obsData.shape)
            obsData[~mask] = 0
            obsDataIVM[~mask] = 0
            # obsData.mask |= ~mask
            # obsDataIVM[~mask] = 0
        # TODO: Use slice to fit only the masked area. But messes up xy pos.
        else:
            warn('pyregion could not be imported. mask_file will be ignored.')

    mc_model = multicomponent_model(obsData, obsDataIVM, psfData, psfDataIVM,
                                    components=model_file,
                                    mag_zp=mag_zeropoint)
    sampler = MCMC(mc_model, db='pickle', name=output_name.format('db'))
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    # Write mean model output files
    obsHeader = pyfits.getheader(obs_file, ignore_missing_end=True)
    write_mean_model(sampler, sampler.db, basename=output_name,
                     filetypes=write_fits, header=obsHeader)
    # TODO: Return something? Maybe model, resid, IVM arrays?


def write_mean_model(model, db, basename='mcmc', filetypes=('residual', ),
                     trace_slice=slice(0, -1), header=None):
    if header is None:
        header = pyfits.Header()
    if '{}' not in basename:
        basename += '_{}'

    # TODO: Strictly speaking, there may be stochastics whose traces are not
    # retained in the database. What to do about those?
    stoch_names = [stoch.__name__ for stoch
                   in model.stochastics - model.observed_stochastics]
    statscards = _stats_as_header_cards(db, trace_names=stoch_names,
                                        trace_slice=trace_slice)
    header.extend(statscards)

    # Set model stochastic values to their trace means
    for stoch in model.stochastics - model.observed_stochastics:
        stoch.value = np.mean(db.trace(stoch.__name__)[trace_slice], axis=0)

    # Save out requested file types
    for out_type in filetypes:
        node_name = out_type
        try:
            outputData = np.ma.filled(model.get_node(node_name).value, 0)
        except AttributeError:
            warn(('Unable to find model parameter named {}. No output will ' +
                  'be written for file type {}').format(node_name, out_type))
            continue

        pyfits.writeto(basename.format(out_type + '.fits'),
                       outputData.copy(), header=header,
                       clobber=True, output_verify='fix')


def _stats_as_header_cards(db, trace_names=None, trace_slice=slice(0, -1)):
    # TODO: better way to make keys. Maybe component.shortname(attr) etc.
    replace_pairs = (('_Sersic', 'SER'), ('_PSF', 'PSF'), ('_Sky', 'SKY'),
                     ('_reff', '_RE'), ('_index', '_N'), ('_axis_ratio', '_Q'),
                     ('_angle', '_ANG'))
    statscards = []
    for trace_name in sorted(trace_names):
        trace = db.trace(trace_name)[trace_slice]
        mean = np.mean(trace, axis=0)
        std = np.std(trace, axis=0)
        key = trace_name
        for oldstr, newstr in replace_pairs:
            key = key.replace(oldstr, newstr)
        try:
            val = '{:0.2f} +/- {:0.2f}'.format(mean, std)
        except ValueError:
            strmean = ','.join(['{:0.2f}'.format(dim) for dim in mean])
            strstd = ','.join(['{:0.2f}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(key, val, 'psfMC model component')]
    return statscards
