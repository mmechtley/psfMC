from __future__ import division, print_function

from warnings import warn

import numpy as np
from astropy.io import fits

from ..database import row_to_param_vector
from ..array_utils import print_progress

default_filetypes = ('raw_model', 'convolved_model', 'composite_ivm',
                     'residual', 'point_source_subtracted')


def save_posterior_images(model, database, output_name='out_{}',
                          mode='weighted', filetypes=default_filetypes,
                          bad_px_value=0):
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
    """
    header = model.obs_header
    if '{}' not in output_name:
        output_name += '_{}'

    statscards = _stats_as_header_cards(model, database)
    header.extend(statscards, end=True)

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


def _stats_as_header_cards(model, database):
    """
    Collates statistics about the trace database, and returns them in 3-tuple
    key-value-comment format suitable for extending a fits header
    """
    # First get information about the sampler run parameters
    stats_cards = _fits_section_header('psfMC MCMC SAMPLER PARAMETERS')
    # Check sampler convergence
    stats_cards += [
        ('MCITER', database.meta['MCITER'], 'number of retained samples'),
        ('MCBURN', database.meta['MCBURN'],
         'number of burn-in (discarded) samples'),
        ('MCCHAINS', database.meta['MCCHAINS'], 'number of chains run'),
        ('MCCONVRG', database.meta['MCCONVRG'], 'Has MCMC sampler converged?'),
        ('MCACCEPT', database.meta['MCACCEPT'],
         'Acceptance fraction (avg of all walkers)')
    ]

    # Now collect information about the posterior model
    stats_cards += _fits_section_header('psfMC POSTERIOR MODEL INFORMATION')
    best_row = np.argmax(database['lnprobability'])
    best_chain = database['walker'][best_row]
    best_sample = best_row % database.meta['MCITER']
    stats_cards += [
        ('MAPCHAIN', best_chain, 'Chain index of maximum posterior model'),
        ('MAPSAMP', best_sample, 'Sample index of maximum posterior model')]

    stoch_col_names = model.param_names
    stoch_fits_abbrs = model.param_fits_abbrs

    for col_name, fits_abbr in zip(stoch_col_names, stoch_fits_abbrs):
        mean_post = np.mean(database[col_name], axis=0)
        std_post = np.std(database[col_name], axis=0)
        try:
            val = '{:0.4g} +/- {:0.4g}'.format(mean_post, std_post)
        except ValueError:
            strmean = ','.join(['{:0.4g}'.format(dim) for dim in mean_post])
            strstd = ','.join(['{:0.4g}'.format(dim) for dim in std_post])
            val = '({}) +/- ({})'.format(strmean, strstd)
        stats_cards += [(fits_abbr, val, 'psfMC model component')]

    # Record the name of the PSF file used in the MP model
    psf_selector = model.config.psf_selector
    if len(psf_selector.psf_list) > 1:
        psf_col = psf_selector.psf_index.name
        best_psf_index = database[psf_col][best_row]
        psf_selector.set_stochastic_values(np.array([best_psf_index]))
    stats_cards += [('PSF_IMG', psf_selector.filename,
                    'PSF image of maximum posterior model')]

    return stats_cards


def _fits_section_header(section_name):
    """
    Blank fits header cards for a section header. As in drizzle, one blank line,
    a line with the section name as a comment, then one more blank.
    """
    return [('', '', ''), ('', '/ ' + section_name, ''), ('', '', '')]
