import numpy as np
from astropy.table import Table


# TODO: decide whether this should be hdf5, or FITS or something else
def save_database(sampler, model, db_name):
    """
    Save database file to disk using astropy Table format

    :param sampler: emcee Sampler object containing sampled chains
    :param model: psfMC MultiComponentModel for getting trace names, etc.
    :param db_name: filename to save to
    :return: database as saved, as astropy Table object
    """
    chain = sampler.chain
    lnprobability = sampler.lnprobability
    stochastic_names = model.param_names
    stochastic_lens = model.param_lens

    flat_chain_shape = (chain.shape[0] * chain.shape[1], chain.shape[2])
    split_inds = np.cumsum(stochastic_lens)[:-1]
    data_cols = np.split(chain.reshape(flat_chain_shape), split_inds, axis=1)

    walker_col = np.repeat(np.arange(chain.shape[0], dtype=int), chain.shape[1])
    stochastic_names += ['lnprobability', 'walker']
    data_cols += [lnprobability.flat, walker_col]

    db = Table(data_cols, names=stochastic_names)
    # TODO: annotate with additional information? random state? burn?

    db.write(db_name, format='hdf5', path='/Chains', overwrite=True)
    return db


def load_database(db_name):
    """
    Load database from disk

    :param db_name: database filename
    :return: astropy Table object
    """
    return Table.read(db_name, format='hdf5', path='/Chains')


def get_sampler_state(database):
    """
    Get the current state of the sampler (last samples from each chain) e.g. in
    order to resume sampling
    :param database: astropy Table as returned by either save_database or
        load_database
    :return: tuple of current walker pos (nwalkers, nparams), lnprobability
    """
    stochastic_cols = database.colnames
    stochastic_cols.remove('walker')
    stochastic_cols.remove('lnprobability')

    nwalkers = database['walker'].max() + 1
    iterations = len(database) / nwalkers
    current_rows = np.arange(nwalkers)*iterations - 1

    walker_pos = database[stochastic_cols][current_rows]
    walker_pos = walker_pos.as_array().view(np.float64)
    nparams = walker_pos.size / nwalkers

    walker_pos = walker_pos.reshape(nwalkers, nparams)
    ln_prob = database['walker'][current_rows].as_array().view(np.float64)

    # TODO: technically should return random_state also
    return walker_pos, ln_prob
