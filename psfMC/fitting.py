from __future__ import division, print_function

from warnings import warn
from emcee import EnsembleSampler
from .database import save_database
from .analysis import check_convergence_autocorr
from .analysis.images import default_filetypes
from .models import MultiComponentModel


def model_galaxy_mcmc(model_file, output_name=None,
                      write_fits=default_filetypes,
                      iterations=0, burn=0,
                      chains=None, max_iterations=1,
                      convergence_check=check_convergence_autocorr,
                      backend='pickle',
                      **kwargs):
    """
    Model the light distribution of a galaxy or galaxies using multi-component
    Markov Chain Monte Carlo parameter estimation.

    :param model_file: Filename of the model definition file. This should be
        a series of components from psfMC.ModelComponents, with parameters
        supplied as either fixed values or stochastics from psfMC.distributions
    :param output_name: Base name for output files (no file extension). By
        default, files are written out containing the raw model, convolved
        model, combined IVM 1/(1/obsIVM + 1/modelIVM), residual of
        observation - model, and the MCMC trace database.
    :param write_fits: List of which fits file types to write. By default, raw
        (unconvolved) model, convolved model, model IVM, and residual.
    :param chains: Number of individual chains to run (resetting the sampler
        to begin at a new random position between each chain)
    :param max_iterations: Maximum sampler iterations before convergence is
        enforced. Default is 1, which means sampler halts even if not converged.
    :param convergence_check: Function taking an MCMC model and a list of chain
        indexes as arguments, and returning True or False based on whether the
        model has converged. Default function returns True when all traces have
        potential scale reduction factor within 0.05 of 1.0. Sampling will be
        repeated (increasing effective burn-in period) until convergence check
        is met or until max_iterations iterations are performed
    :param backend: PyMC database backend to use. pickle is default due to
        universal support, but HDF5 (via pytables) is highly recommended
    :param kwargs: Further keyword arguments are passed to pyMC.MCMC.sample, and
        can be used to control number of MCMC samples, burn-in period, etc.
        Useful parameters include iter=, burn=, tune_interval=, thin=, etc. See
        pyMC documentation.
    """
    if output_name is None:
        output_name = 'out_' + model_file.replace('.py', '')
    output_name += '_{}'

    mc_model = MultiComponentModel(components=model_file)

    # If chains is not specified, use the minimum number recommended by emcee
    if chains is None:
        chains = 2 * mc_model.num_params + 2

    # TODO: can't use threads=n right now because model object can't be pickled
    sampler = EnsembleSampler(nwalkers=chains, dim=mc_model.num_params,
                              lnpostfn=mc_model.log_posterior,
                              kwargs={'model': mc_model})

    # Open database if it exists, otherwise pass backend to create a new one
    db_name = output_name.format('db.hdf5')

    # TODO: Check if database exists, resume if so
    # else:
    # warn('Database already contains sampled chains, skipping sampling')

    param_vec = mc_model.init_params_from_priors(chains)

    # Run burn-in and discard
    for step, (param_vec, logp, rand_state) in enumerate(
            sampler.sample(param_vec, iterations=burn)):
        next_pct = 100 * (step + 1) // burn
        curr_pct = 100 * step // burn
        if next_pct - curr_pct > 0:
            print('Burning: {:d}%'.format(next_pct))

    sampler.reset()

    for sampling_iter in range(max_iterations):

        # Now run real samples and retain
        for step, result in enumerate(
                sampler.sample(param_vec, iterations=iterations)):
            next_pct = 100 * (step + 1) // iterations
            curr_pct = 100 * step // iterations
            if next_pct - curr_pct > 0:
                print('Sampling: {:d}%'.format(next_pct))

        # TODO: Replace with user-defined convergence check


        if convergence_check(sampler):
            break
        else:
            warn('Not yet converged after {:d} iterations'
                 .format((sampling_iter + 1)*iterations))

    db = save_database(sampler, mc_model, db_name)

    # Write model output files, using only the last "chains" chains.
    # post_chains = range(mc_model.db.chains - chains, mc_model.db.chains)
    # save_posterior_images(mc_model, output_name=output_name,
    #                       filetypes=write_fits,
    #                       chains=post_chains,
    #                       convergence_check=convergence_check)
