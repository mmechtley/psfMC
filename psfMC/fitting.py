from __future__ import division
import pyfits
import numpy as np
from warnings import warn
from pymc.MCMC import MCMC
from .models import multicomponent_model
try:
    import pyregion
except ImportError:
    pyregion = None


# TODO: Friendlier interface for supplying components. File?
def model_galaxy_mcmc(obs_file, obsIVM_file, psf_file, psfIVM_file,
                      fit_components=None, mag_zeropoint=0,
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
    :param fit_components: List of tuples, each defining a component for the
        fit. First element of each tuple is a string specifying the component
        type (psf, sersic). Following elements specify the min and max values
        for each parameter. See documentation for different component types and
        parameter orders.
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
    if fit_components is None:
        raise ValueError('No fitting components specified. Please supply at ' +
                         'least one component.')
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

    if mask_file is not None:
        if pyregion is not None:
            hdr = pyfits.getheader(obs_file)
            regfilt = pyregion.open(mask_file).as_imagecoord(hdr).get_filter()
            mask = regfilt.mask(obsData.shape)
            obsData = np.ma.masked_array(obsData, mask=~mask)
            obsDataIVM[~mask] = 0
            # TODO: Use slice to fit only the masked area. But messes up xy pos.
        else:
            warn('pyregion could not be imported. mask_file will be ignored.')

    # Normalize the PSF kernel
    # TODO: Convert to float64 first?
    psfScale = 1 / psfData.sum()
    psfData *= psfScale
    psfDataIVM /= psfScale**2

    mc_model = multicomponent_model(obsData, obsDataIVM, psfData, psfDataIVM,
                                    components=fit_components,
                                    magZP=mag_zeropoint)
    sampler = MCMC(mc_model, db='pickle', name=output_name.format('db'))
    sampler.sample(**kwargs)

    ## Saves out to pickle file
    sampler.db.close()

    # TODO: Remove debug output
    stats = sampler.stats()
    for stoch in sorted(stats):
        if stoch.startswith(tuple(str(i) for i in range(len(fit_components)))):
            print '{}: mean: {} std: {}'.format(
                stoch, stats[stoch]['mean'], stats[stoch]['standard deviation'])

    # TODO: Add fit information to fits headers
    obsHeader = pyfits.getheader(obs_file, ignore_missing_end=True)
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