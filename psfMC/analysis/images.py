from __future__ import division, print_function

from warnings import warn
from collections import OrderedDict

import numpy as np
from astropy.io import fits

from ..database import row_to_param_vector, annotate_metadata, \
    filter_lowp_walkers
from ..utils import print_progress

default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                     'residual', 'point_source_subtracted')


def save_posterior_images(model, database, output_name='out_{}',
                          mode='weighted', filetypes=default_filetypes,
                          bad_px_value=0, stats_min_percentile=10):
    """
    Writes out the posterior model images. Two modes are supported: Maximum a
    posteriori (maximum or MAP) and "weighted average" (weighted). Since
    each MCMC trace represents (ideally) unbiased draws from the posterior
    distribution, the weighted images are simply the (per-pixel) mean of all
    sample images. The "weighting" is handled by the fact that more probable
    locations in parameter space will be more densely populated with samples.
    Sample autocorrelation should be moot, since such correlations are
    distributed roughly equally in parameter space.

    :param model: MultiComponentModel that defines the model.
    :param database: astropy Table holding sampler database information
    :param output_name: base name for output files (no extension)
    :param mode: Either "maximum" or "MAP" for MAP, or "weighted" (Default)
    :param filetypes: list of filetypes to save out (see model_galaxy_mcmc
        documentation for a list of possible types
    :param bad_px_value: Value to replace bad pixels with
    :param stats_min_percentile: filter out stuck walkers whose probabilities
        are all below given percentile when reporting stats in header
        (default: 10%)
    """
    header = model.obs_header
    if '{}' not in output_name:
        output_name += '_{}'

    _add_stats_to_header(header, model, database, stats_min_percentile)

    print('Saving posterior models')
    # Check to ensure we understand all the requested file types
    unknown_ftypes = set(ftype for ftype in filetypes
                         if not hasattr(model, ftype))
    if len(unknown_ftypes) != 0:
        warn('Unknown filetypes requested: {} Output images will not be '
             'generated for these types.'.format(unknown_ftypes))
        filetypes = set(filetypes) - unknown_ftypes

    output_data = dict()
    stochastic_cols = tuple(model.param_names)
    if mode in ('maximum', 'MAP'):
        best_row = np.argmax(database['lnprobablity'])
        best_row = database[stochastic_cols][best_row]
        model.param_values = row_to_param_vector(best_row)

        for ftype in filetypes:
            sample_data = getattr(model, ftype)()
            sample_data[~np.isfinite(sample_data)] = bad_px_value
            output_data[ftype] = sample_data

    elif mode in ('weighted',):
        total_samples = len(database)
        # Only re-calculate images if we loaded a database without sampling.
        # Otherwise, model has already accumulated weighted images during
        # sampling
        if total_samples != model.accumulated_samples:
            model.reset_images()
            for sample, row in enumerate(database[stochastic_cols]):
                print_progress(sample, total_samples,
                               'Creating posterior images')
                model.param_values = row_to_param_vector(row)

                sample_imgs = {ftype: getattr(model, ftype)()
                               for ftype in filetypes}
                model.accumulate_images([sample_imgs])

        for ftype in filetypes:
            out_img = model.posterior_images[ftype]
            out_img[~np.isfinite(out_img)] = bad_px_value
            output_data[ftype] = out_img

    else:
        warn('Unknown posterior output mode ({}). Posterior model images will '
             'not be saved.'.format(mode))
        return

    # Now save the files
    for ftype in filetypes:
        header.set('OBJECT', value=ftype)
        fits.writeto(output_name.format(ftype) + '.fits',
                     output_data[ftype], header=header,
                     clobber=True, output_verify='fix')
    return


def _add_stats_to_header(header, model, database, filter_percentile):
    """
    Collates statistics about the trace database, and adds them to the supplied
    FITS header
    """
    # First add all extra metadata from the trace database
    header.extend(_fits_section_header('psfMC MCMC SAMPLER PARAMETERS'))

    database_info = annotate_metadata(database.meta)
    header.update(database_info)

    # Now collect information about the posterior model
    header.extend(_fits_section_header('psfMC POSTERIOR MODEL INFORMATION'))

    model_stats = OrderedDict()
    stoch_col_names = model.param_names
    stoch_fits_abbrs = model.param_fits_abbrs

    filtered_database = filter_lowp_walkers(
        database, percentile=filter_percentile)

    for col_name, fits_abbr in zip(stoch_col_names, stoch_fits_abbrs):
        mean_post = np.mean(filtered_database[col_name], axis=0)
        std_post = np.std(filtered_database[col_name], axis=0)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(mean_post, std_post)
        except (ValueError, TypeError):
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in mean_post])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std_post])
            val = '({}) +/- ({})'.format(strmean, strstd)
        model_stats[fits_abbr] = val

    # Record the name of the PSF file used in the MP model
    psf_selector = model.config.psf_selector
    if len(psf_selector.psf_list) > 1:
        psf_col = 'PSF_Index'
        best_psf_index = database[psf_col][header['MAPROW']]
        psf_selector.set_stochastic_values(np.array([best_psf_index]))
    model_stats['PSFIMG'] = psf_selector.filename

    model_stats = annotate_metadata(model_stats)
    header.update(model_stats)
    return


def _fits_section_header(section_name):
    """
    Blank fits header cards for a section header. As in drizzle, one blank line,
    a line with the section name as a comment, then one more blank.
    """
    return [('', '', ''), ('', '/ ' + section_name, ''), ('', '', '')]
