from __future__ import division
from warnings import warn

import pyfits
import numpy as np
from pymc.MCMC import MCMC

from .models import multicomponent_model
from .model_parser import component_list_from_file
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
    try:
        fit_components = component_list_from_file(model_file)
    except IOError, err:
        message = 'Unable to open components file {}. Does it exist?'
        err.message = message.format(model_file)
        raise err

    if output_name is None:
        output_name = obs_file.replace('.fits', '')
    output_name += '_{}'

    # TODO: Set these based on total number of unknown components
    kwargs.setdefault('iter', 6000)
    kwargs.setdefault('burn', 3000)
    kwargs.setdefault('tune_interval', 25)
    kwargs.setdefault('thin', 5)

    obsData = pyfits.getdata(obs_file, ignore_missing_end=True)
    obsDataIVM = pyfits.getdata(obsIVM_file, ignore_missing_end=True)
    psfData = pyfits.getdata(psf_file, ignore_missing_end=True)
    psfDataIVM = pyfits.getdata(psfIVM_file, ignore_missing_end=True)

    obsData, obsDataIVM, psfData, psfDataIVM = mask_bad_pixels(
        obsData, obsDataIVM, psfData, psfDataIVM)

    # FIXME: Masks are breaking fitting
    # if mask_file is not None:
    #     if pyregion is not None:
    #         hdr = pyfits.getheader(obs_file)
    #         regfilt = pyregion.open(mask_file).as_imagecoord(hdr).get_filter()
    #         mask = regfilt.mask(obsData.shape)
    #         obsData.mask |= ~mask
    #         obsDataIVM[~mask] = 0
        # TODO: Use slice to fit only the masked area. But messes up xy pos.
        # else:
        #     warn('pyregion could not be imported. mask_file will be ignored.')

    # Normalize the PSF kernel
    # TODO: Convert to float64 first?
    psf_sum = psfData.sum()
    psfData /= psf_sum
    psfDataIVM *= psf_sum**2

    mc_model = multicomponent_model(obsData, obsDataIVM, psfData, psfDataIVM,
                                    components=fit_components,
                                    mag_zp=mag_zeropoint)
    sampler = MCMC(mc_model, db='pickle', name=output_name.format('db'))
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    stats = sampler.stats()
    statscards = _to_header_cards(stats)

    obsHeader = pyfits.getheader(obs_file, ignore_missing_end=True)
    obsHeader.extend(statscards)
    for out_type in write_fits:
        node_name = out_type
        if out_type == 'residual':
            node_name = 'convolved_model'
        # TODO: Is get_node the best way to get at non-traced model data?
        try:
            outputData = np.ma.filled(sampler.get_node(node_name).value, 0)
        except AttributeError:
            warn(('Unable to find model parameter named {}. No output will ' +
                  'be written for file type {}').format(node_name, out_type))
            continue
        if out_type == 'residual':
            obsData = pyfits.getdata(obs_file, ignore_missing_end=True)
            outputData = obsData - outputData

        pyfits.writeto(output_name.format(out_type + '.fits'),
                       outputData.copy(), header=obsHeader,
                       clobber=True, output_verify='fix')

    # TODO: Return something? Maybe model, resid, IVM arrays?


def _to_header_cards(stats):
    # TODO: better way to make keys. Maybe component.shortname(attr) etc.
    replace_pairs = (('_Sersic','SER'), ('_PSF','PSF'), ('_Sky', 'SKY'),
                     ('_reff', '_RE'), ('_index', '_N'), ('_axis_ratio','_Q'),
                     ('_angle', '_ANG'))
    statscards = []
    print sorted(stats)
    for stoch in sorted(stats):
        mean = stats[stoch]['mean']
        std = stats[stoch]['standard deviation']
        key = stoch
        for oldstr, newstr in replace_pairs:
            key = key.replace(oldstr, newstr)
        try:
            val = '{:0.2f} +/- {:0.2f}'.format(mean, std)
        except ValueError:
            strmean = ','.join(['{:0.2f}'.format(dim) for dim in mean])
            strstd = ','.join(['{:0.2f}'.format(dim) for dim in std])
            val = '({}) +/- ({})'.format(strmean, strstd)
        statscards += [(key, val, 'psfMC model component')]
        # TODO: Remove debug output
        print '{}: {}'.format(stoch, val)
    return statscards
