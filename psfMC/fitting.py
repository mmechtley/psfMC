from __future__ import division, print_function
import os
from collections import OrderedDict
from warnings import warn
from emcee import EnsembleSampler
from .array_utils import print_progress
from .analysis import check_convergence_autocorr, save_posterior_images
from .analysis.images import default_filetypes
from .database import save_database, load_database
from .models import MultiComponentModel


def model_galaxy_mcmc(model_file, output_name=None,
                      write_fits=default_filetypes,
                      iterations=0, burn=0,
                      chains=None, max_iterations=1,
                      convergence_check=check_convergence_autocorr):
    """
    Model the surface brightness distribution of a galaxy or galaxies using
    multi-component Markov Chain Monte Carlo parameter estimation.

    :param model_file: Filename of the model definition file. This should be
        a series of components from psfMC.ModelComponents, with parameters
        supplied as either fixed values or stochastics from psfMC.distributions
    :param output_name: Base name for output files (no file extension). By
        default, files are written out containing the requested image types
        (write_fits param) and the MCMC trace database. If None, use
        out_<model_filename>
    :param write_fits: List of which fits file types to write. By default, raw
        (unconvolved) model, convolved model, model IVM, residual, and point
        sources subtracted.
    :param iterations: Number of retained MCMC samples
    :param burn: Number of discarded (burn-in) MCMC samples
    :param chains: Number of individual chains (walkers) to run. If None, the
        minimum number recommended by emcee will be used. More is better.
    :param max_iterations: Maximum sampler iterations before convergence is
        enforced. Default is 1, which means sampler halts even if not converged.
    :param convergence_check: Function taking an emcee Sampler as argument, and
        returning True or False based on whether the sampler has converged.
        Default function returns True when the autocorrelation time of all
        stochastic variables is < 10% of the total number of samples. Sampling
        will be repeated (increasing the chain length) until convergence check
        is met or until max_iterations iterations are performed.
    """
    if output_name is None:
        output_name = 'out_' + model_file.replace('.py', '')
    output_name += '_{}'

    mc_model = MultiComponentModel(components=model_file)

    # If chains is not specified, use the minimum number recommended by emcee
    if chains is None:
        chains = 2 * mc_model.num_params + 2

    # FIXME: can't use threads=n right now because model object can't be pickled
    sampler = EnsembleSampler(nwalkers=chains, dim=mc_model.num_params,
                              lnpostfn=mc_model.log_posterior,
                              kwargs={'model': mc_model})

    # Open database if it exists, otherwise pass backend to create a new one
    db_name = output_name.format('db') + '.fits'

    # TODO: Resume if database exists
    if not os.path.exists(db_name):
        param_vec = mc_model.init_params_from_priors(chains)

        # Run burn-in and discard
        for step, result in enumerate(
                sampler.sample(param_vec, iterations=burn)):
            # Set new initial sampler state
            param_vec = result[0]
            # No need to retain images from every step, so clear blobs
            sampler.clear_blobs()
            print_progress(step, burn, 'Burning')

        sampler.reset()

        converged = False
        for sampling_iter in range(max_iterations):
            # Now run real samples and retain
            for step, result in enumerate(
                    sampler.sample(param_vec, iterations=iterations)):
                mc_model.accumulate_images(result[3])
                # No need to retain images from every step, so clear blobs
                sampler.clear_blobs()
                print_progress(step, iterations, 'Sampling')

            if convergence_check(sampler):
                converged = True
                break
            else:
                warn('Not yet converged after {:d} iterations:'
                     .format((sampling_iter + 1)*iterations))
                convergence_check(sampler, verbose=1)

        # Collect some metadata about the sampling process. These will be saved
        # in the FITS headers of both the output database and the images
        db_metadata = OrderedDict([
            ('MCITER', sampler.chain.shape[1]),
            ('MCBURN', burn),
            ('MCCHAINS', chains),
            ('MCCONVRG', converged),
            ('MCACCEPT', sampler.acceptance_fraction.mean())
        ])
        database = save_database(sampler, mc_model, db_name,
                                 meta_dict=db_metadata)
    else:
        print('Database already contains sampled chains, skipping sampling')
        database = load_database(db_name)

    # Write model output files
    save_posterior_images(mc_model, database, output_name=output_name,
                          filetypes=write_fits)
