from __future__ import division, print_function
from warnings import warn
import os
import pymc
import pymc.database
from pymc.StepMethods import AdaptiveMetropolis, DiscreteMetropolis
from .models import multicomponent_model
from .analysis import check_convergence_psrf, save_posterior_images
from .analysis.images import default_filetypes


def model_galaxy_mcmc(model_file, output_name=None,
                      write_fits=default_filetypes,
                      chains=1, max_iterations=1,
                      convergence_check=check_convergence_psrf,
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

    # TODO: Set these based on total number of unknown components?
    kwargs.setdefault('iter', 6000)
    kwargs.setdefault('burn', 3000)

    # Open database if it exists, otherwise pass backend to create a new one
    db_name = output_name.format('db')
    if os.path.exists(db_name+'.'+backend):
        back_module = getattr(pymc.database, backend)
        db = back_module.load(db_name+'.'+backend)
    else:
        db = backend

    mc_model = multicomponent_model(components=model_file, db=db, name=db_name)

    # TODO: Add support for resuming. For now, skip sampling if chains exist
    if mc_model.db.chains == 0:
        _set_step_methods(mc_model)

        for samp_iter in range(max_iterations):
            # TODO: Is there a way to delete old chains?
            for chain_num in range(chains):
                # Seed new values for every independent chain on first iteration
                # On subsequent iterations, load last sample from previous
                if samp_iter == 0:
                    mc_model.draw_from_prior()
                else:
                    mc_model.remember(chain=(samp_iter-1)*chain_num,
                                      trace_index=-1)
                mc_model.sample(**kwargs)

            iter_chains = range(samp_iter*chains, (samp_iter+1)*chains)
            if convergence_check(mc_model, chains=iter_chains):
                break
            else:
                warn('Not yet converged after {:d} iterations ({:d} chains)'
                     .format(samp_iter+1, chains))
        mc_model.db.commit()
    else:
        warn('Database already contains sampled chains, skipping sampling')

    # Write model output files, using only the last "chains" chains.
    post_chains = range(mc_model.db.chains - chains, mc_model.db.chains)
    save_posterior_images(mc_model, output_name=output_name,
                          filetypes=write_fits,
                          chains=post_chains,
                          convergence_check=convergence_check)
    mc_model.db.close()


def _set_step_methods(model):
    """
    Set special step methods (for xy positions and discrete variables)
    :param model: multicomponent model
    """
    for stoch in model.step_method_dict:
        if 'xy' in stoch.__name__:
            model.use_step_method(AdaptiveMetropolis, stoch)
        if stoch.__name__ == 'PSF_Index':
            model.use_step_method(DiscreteMetropolis, stoch,
                                  proposal_distribution='Prior')
