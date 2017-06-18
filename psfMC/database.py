import numpy as np
from collections import OrderedDict
from astropy.table import Table


def save_database(sampler, model, db_name, meta_dict=None):
    """
    Save database file to disk using astropy Table format

    :param sampler: emcee Sampler object containing sampled chains
    :param model: psfMC MultiComponentModel for getting trace names, etc.
    :param db_name: filename to save to
    :param meta_dict: OrderedDict of additional information to save in the FITS
        header (e.g. sampler arguments, burn-in length, etc.) Dict keys must be
        short enough to be valid FITS header keys
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
    # TODO: Figure out how to save random state. Second table?
    if meta_dict is None:
        meta_dict = OrderedDict()
    meta_dict['MAPROW'] = np.argmax(db['lnprobability'])
    meta_dict = annotate_metadata(meta_dict)
    db.meta.update(meta_dict)

    db.write(db_name, format='fits', overwrite=True)
    # FIXME: Should not need to reload the db we just created, but posterior
    # model creation was failing (something with the way db Rows were translated
    # to ndarray vector)
    return load_database(db_name)


def load_database(db_name):
    """
    Load database from disk

    :param db_name: database filename
    :return: astropy Table object
    """
    return Table.read(db_name, format='fits')


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


def row_to_param_vector(table_row):
    row_vec = table_row.as_void()
    new_dtype = row_vec.dtype[0]
    return np.frombuffer(row_vec.data, dtype=new_dtype)


def annotate_metadata(input_dict):
    """
    Take input metadata (OrderedDict) and annotate with appropriate FITS header
    comments.
    """
    comments = {'MCITER': 'number of retained samples',
                'MCBURN': 'number of burn-in (discarded) samples',
                'MCWALKRS': 'number of walkers run',
                'MCCONVRG': 'Has MCMC sampler converged?',
                'MCACCEPT': 'Acceptance fraction (avg of all walkers)',
                'MAPROW': 'Row index of maximum posterior model',
                'PSFIMG': 'PSF image of maximum posterior model'}
    output_dict = input_dict.copy()
    for key in input_dict.keys():
        # If we encounter an unknown key, assume it's a model parameter
        output_dict[key] = (input_dict[key],
                            comments.get(key, 'psfMC model parameter'))

    return output_dict


def filter_lowp_walkers(database, percentile=10):
    """
    Filter out (remove) entries for walkers that never enter the
    high-probability region. These are usually "lost" walkers that start out in
    very low-p regions and never make it out.
    :param database: Database table
    :param percentile: Walkers will be discarded if ALL of their samples have
        lnprobability below percentile.
    :return: Filtered database
    """
    # Filter probabilities one-sided (low-p side only)
    pct_value = np.percentile(database['lnprobability'], percentile)
    ok_walkers = np.unique(
        database['walker'][database['lnprobability'] > pct_value])
    return database[np.in1d(database['walker'], ok_walkers)]