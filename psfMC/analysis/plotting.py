# coding=utf-8
from __future__ import division, print_function
import os
from warnings import warn
import pymc.database
import matplotlib.pyplot as pp
import numpy as np
from matplotlib import patheffects
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import gamma
from .statistics import num_effective_samples, potential_scale_reduction, \
    max_posterior_sample, chain_autocorr
from ..models import multicomponent_model
from ..ModelComponents.Sersic import Sersic
from corner import corner

_labels = {'deviance': 'Model deviance (-2*log(P))',
           'x': '{} x (pix)',
           'y': '{} y (pix)',
           'adu': '{} (adu)',
           'mag': '{} mag',
           'index': '{} index $n$',
           'reff': '{} $R_e a$ (pix)',
           'reff_b': '{} $R_e b$ (pix)',
           'angle': '{} PA (deg)',
           'PSF_Index': 'PSF index',
           'axisratio': '{} axis ratio $b/a$',
           'sbeff': '{} $\mu_e$ (mag pixel$^{-1}$)',
           'magdiff': '$m_{{{}}} - m_{{{}}}$',
           'centerdist': '{} vs. {} position difference (pixels)'}

_text_stroke = patheffects.withStroke(linewidth=3, foreground='w')


def _axis_label(trace_name):
    """
    Returns an axis label for the specified trace name
    """
    if trace_name in _labels.keys():
        return _labels[trace_name]
    elif '_' in trace_name:
        # Extract 2 underscore-separated elements (# + component type) at a
        # time. Stop when the leftover bits are a known key, or empty string.
        comps = []
        while trace_name not in _labels.keys() and trace_name != '':
            index, comp, trace_name = trace_name.split('_', 2)
            comps += [u'({}) {}'.format(index, comp)]
        return _labels.get(trace_name, trace_name).format(*comps)
    else:
        return trace_name


def _get_prior(trace_name, model):
    """
    Return a function P(X) representing the prior probability for key. For keys
    that don't have priors (e.g. centerdist, magdiff, etc.), returns None.
    """
    key_node = model.get_node(trace_name)
    if key_node is None:
        return None
    prior_logp = key_node.__dict__['_logp'].fun
    prior_args = dict(key_node.__dict__['_logp'].arguments)
    del prior_args['value']

    def prior_func(x_vals):
        return np.exp([prior_logp(value=val, **prior_args) for val in x_vals])

    return prior_func


def _select_chains(db, chains):
    """
    Select database chain indexes. Allows for negative indexes (i.e. last chain,
    second-to-last chain, etc.).
    """
    if chains is None:
        return range(db.chains)
    else:
        return np.arange(db.chains)[chains]


def _sbeff(mag, reff, reff_b, index, px_scale=None):
    """
    Sersic profile surface brightness at effective radius r_e.
    """
    # TODO: Remove this since it duplicates code in Sersic class.
    # Need to make it static there though.
    kappa = Sersic.kappa(index)
    to_magarc = 2.5*np.log10(px_scale**2) if px_scale is not None else 0
    return mag + 2.5*np.log10(2 * np.pi * reff * reff_b * index *
                              np.exp(kappa) / kappa**(2*index) *
                              gamma(2*index)) + to_magarc


def _get_trace(trace_name, db, chain=1):
    """
    Get the trace array for the specified parameter key. Also handles lookup and
    calculation of the special keys magdiff, centerdist, axisratio, and sbeff.
    e.g. 1_PSF_2_Sersic_centerdist or 2_Sersic_sbeff
    :param trace_name: Name of the model parameter to get the trace for
    :param db: Pymc database object to get trace from
    :param chain: Chain index to get trace from
    :return: Trace values as NxD array, where N is the number of samples and D
        is the number of dimensions of the parameter (2 for xy, 1 for others)
    """
    try:
        name_comps = trace_name.split('_')
        if 'magdiff' in name_comps:
            key_mag1 = '_'.join(name_comps[0:2]+['mag'])
            key_mag2 = '_'.join(name_comps[2:4]+['mag'])
            trace = (db.trace(key_mag1, chain)[:] -
                     db.trace(key_mag2, chain)[:])
        elif 'centerdist' in name_comps:
            key_xy1 = '_'.join(name_comps[0:2]+['xy'])
            key_xy2 = '_'.join(name_comps[2:4]+['xy'])
            cdiff = (db.trace(key_xy1, chain)[:] -
                     db.trace(key_xy2, chain)[:])
            trace = np.sqrt(np.sum(cdiff**2, axis=1))
        elif 'axisratio' in name_comps:
            key_prefix = '_'.join(name_comps[0:2]+[''])
            trace = (db.trace(key_prefix+'reff_b', chain)[:] /
                     db.trace(key_prefix+'reff', chain)[:])
        elif 'sbeff' in name_comps:
            key_prefix = '_'.join(name_comps[0:2]+[''])
            trace = _sbeff(db.trace(key_prefix+'mag', chain)[:],
                           db.trace(key_prefix+'reff', chain)[:],
                           db.trace(key_prefix+'reff_b', chain)[:],
                           db.trace(key_prefix+'index', chain)[:])
        else:
            trace = db.trace(trace_name, chain)[:]
    except KeyError as err:
        names = db.trace_names[chain]
        err.message = 'Unable to find trace {} while plotting {}. Available '\
            'traces are {} or magdiff, centerdist, axisratio, sbeff'\
            .format(err, trace_name, names)
        raise err

    # For 1D traces (all except XY coordinates), expand to Nx1 2D array
    if len(trace.shape) == 1:
        trace = np.expand_dims(trace, 1)
    return trace


def _load_db_and_model(db, model):
    """
    Open database file and model file, returning display name (database name
    without extension), pymc database object, and pymc model object.
    """
    # TODO: Allow already-opened database and model, make public
    disp_name, ext = os.path.splitext(os.path.basename(db))
    db_module = getattr(pymc.database, ext[1:])
    db = db_module.load(db)

    if model is not None:
        try:
            model = multicomponent_model(model)
        except IOError:
            print('Unable to find model file {}. Priors will not be plotted.'
                  .format(model))
            model = None
    return disp_name, db, model


def plot_trace(trace_name, db, chains=None, save=False):
    """
    Plot a trace value vs. time (or sample number). Useful for assessing if the
    sampler has reached a stable distribution.
    :param trace_name: Name of traced quantity, including all with priors, as
        well as derived quantities magdiff, centerdist, sbeff, and axisratio.
    :param db: Filename of pymc database
    :param chains: Chain indexes to plot
    :param save: If True, plots will not be displayed but will be saved to disk
        in pdf format.
    """
    disp_name, db, model = _load_db_and_model(db, None)
    chains = _select_chains(db, chains)

    ax_trace = pp.subplot(111)

    divider = make_axes_locatable(ax_trace)
    ax_hist = divider.append_axes('right', size=1.2, pad=0.1, sharey=ax_trace)

    ticker = MaxNLocator(nbins=3, integer=True)
    ax_hist.get_xaxis().set_major_locator(ticker)
    pp.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.get_xaxis().tick_top()

    for chain in chains:
        ml_model, ml_chain = max_posterior_sample(db, [chain])
        trace = _get_trace(trace_name, db, chain)

        for col in range(trace.shape[1]):
            ax_trace.plot(np.arange(trace.shape[0]), trace[:, col])
            ax_hist.hist(trace[:, col], bins=20, histtype='step',
                         orientation='horizontal')
            ml_val = trace[ml_model, col]
            ax_hist.axhline(ml_val, color='Orange', lw=2)

    ax_trace.set_xlabel('Sample')
    ax_trace.set_ylabel(_axis_label(trace_name))
    pp.gcf().suptitle(disp_name)

    if save:
        pp.savefig('_'.join([disp_name, trace_name, 'trace.pdf']))
    else:
        pp.show()
    pp.close(pp.gcf())
    db.close()


# TODO: Split histogram and autocorrelation functions?
def plot_hist(trace_name, db, model=None, chains=None, save=False,
              show_stats=True):
    """
    Plot per-chain histograms for the given traced quantity.
    In addition, plots autocorrelation functions for each chain to help assess
    effective number of samples.
    :param trace_name: Name of traced quantity, including all with priors, as
        well as derived quantities magdiff, centerdist, sbeff, and axisratio.
    :param db: Filename of pymc database
    :param model: Filename of psfMC model. If model is None, priors will not be
        plotted.
    :param chains: Chain indexes to plot
    :param save: If True, plots will not be displayed but will be saved to disk
        in pdf format.
    :param show_stats: If True, convergence statistics will be displayed on the
        plots. Gelman-Rubin Potential Scale Reduction Factor and variance-based
        estimate of effective samples for histogram plot, and Initial Positive
        Sequence estimator of effective samples for autocorrelation plot.
    :return:
    """
    disp_name, db, model = _load_db_and_model(db, model)
    chains = _select_chains(db, chains)

    fig_hist = pp.figure()
    ax_hist = fig_hist.add_subplot(111)

    fig_acorr = pp.figure()
    ax_autocorr = fig_acorr.add_subplot(111)

    all_traces = []
    neff_autocorr = 0
    autocorr_labels = []
    maxlag = 0

    for chain in chains:
        trace = _get_trace(trace_name, db, chain)
        ml_model, ml_chain = max_posterior_sample(db, [chain])

        all_traces.append(trace)
        for col in range(trace.shape[1]):
            chain_color = ax_hist._get_lines.color_cycle.next()
            ax_hist.hist(trace[:, col], bins=20, histtype='step',
                         lw=2, color=chain_color)
            ml_val = trace[ml_model, col]
            ax_hist.axvline(ml_val, color=chain_color, lw=2, ls='dashed')

            lags, corr, eff_samples, chain_maxlag = \
                chain_autocorr(trace[:, col])
            maxlag = np.max([maxlag, chain_maxlag])
            neff_autocorr += eff_samples

            ax_autocorr.plot(lags, corr, color=chain_color,
                             marker=None, ls='solid', lw=2)
            ax_autocorr.axvline(chain_maxlag, color=chain_color)

            autocorr_labels += ['$n_{{eff,{:d}}}$ = {:0.1f}'.format(
                chain, eff_samples)]

    # Calculate PSR factor and effective samples for each dimension
    psr = [potential_scale_reduction([trace[:, col] for trace in all_traces])
           for col in range(all_traces[0].shape[1])]
    neff = [num_effective_samples([trace[:, col] for trace in all_traces])
            for col in range(all_traces[0].shape[1])]

    fig_hist.suptitle(disp_name)
    ax_hist.set_xlabel(_axis_label(trace_name))
    ax_hist.set_ylabel('Number of Samples')

    fig_acorr.suptitle(disp_name)
    ax_autocorr.set_xlim(0, maxlag+10)
    ax_autocorr.axhline(0.0, color='black')
    ax_autocorr.set_xlabel('Lag Length (Samples)')
    ax_autocorr.set_ylabel('Autocorrelation (Normalized)')

    if show_stats:
        # Label for potential scale reduction factors
        psr_fmt = '\n'.join(['$\widehat{{R}}$ = {:0.3f}']*len(psr) +
                            ['$n_{{eff}}$ = {:0.1f}']*len(neff))
        ax_hist.text(0.95, 0.95, psr_fmt.format(*(psr+neff)),
                     ha='right', va='top',  # path_effects=[_text_stroke],
                     transform=ax_hist.transAxes)
        # Label for autocorrelation effective samples
        autocorr_labels += ['$n_{{eff,tot}}$ = {:0.1f}'.format(neff_autocorr)]
        ax_autocorr.text(0.95, 0.95, '\n'.join(autocorr_labels),
                         va='top', ha='right',  # path_effects=[_text_stroke],
                         transform=ax_autocorr.transAxes)

    # Display the prior for this traced variable, if it has a prior
    if model is not None:
        prior_func = _get_prior(trace_name, model)
        if prior_func is not None:
            # TODO: Pick x range where prior has support? (From inverse CDF)
            min_x, max_x = np.min(all_traces), np.max(all_traces)
            mid = 0.5*(min_x + max_x)
            ptp = (max_x - min_x) * 1.1  # 5% higher/lower than min/max
            prior_x = np.linspace(mid-ptp*0.5, mid+ptp*0.5, 100)

            # display prior with normalized (axes) y coordinate
            prior_xform = blended_transform_factory(ax_hist.transData,
                                                    ax_hist.transAxes)
            ax_hist.plot(prior_x, prior_func(prior_x),
                         lw=1, color='black', zorder=-1,
                         transform=prior_xform)

    if save:
        fig_hist.savefig('_'.join([disp_name, trace_name, 'hist.pdf']))
        fig_acorr.savefig('_'.join([disp_name, trace_name, 'acorr.pdf']))
        psr_fmt = '{} mu: {:0.5g} sig: {:0.3g} R: {} ne: {} nea: {}'
        all_traces = np.array(all_traces)
        mean = np.mean(all_traces)
        std = np.std(all_traces)
        print(psr_fmt.format(trace_name, mean, std, psr, neff, neff_autocorr))
    else:
        pp.show()
    pp.close(fig_hist)
    pp.close(fig_acorr)
    db.close()

def corner_plot(database, model=None, disp_parameters=None, chains=None,
                save=False, skip_zero_variance=True, print_stats=True,
                **kwargs):
    """

    :param database:
    :param model:
    :param disp_parameters:
    :param chains:
    :param save:
    :param kwargs:
    :param skip_zero_variance:
    :param print_stats:
    :return:
    """
    disp_name, db, model = _load_db_and_model(database, model)
    chains = _select_chains(db, chains)
    # find available traces and remove deviance/adaptive scale factors
    disp_trace_names = db.trace_names
    if isinstance(disp_trace_names[0], list):
        disp_trace_names = disp_trace_names[0]
    # If no traces were specified, filter scale factors & deviance, then sort
    if disp_parameters is None:
        disp_trace_names = [trace_name for trace_name in disp_trace_names if
                            'adaptive_scale_factor' not in trace_name and
                            trace_name != 'deviance']
        disp_trace_names.sort()
    # If we specified traces, display those ones
    else:
        missing_traces = set(disp_parameters) - set(disp_trace_names)
        if missing_traces != set():
            raise ValueError('Unable to find trace(s) named: {}'.format(
                missing_traces))
        disp_trace_names = list(disp_parameters)

    # Just read in traces as giant data array (samples x parameters)
    # (As expected by triangle.corner())
    flat_traces = None
    for chain in chains:
        traces = [_get_trace(trace_name, db, chain=chain)
                  for trace_name in disp_trace_names]
        flat_chain = np.column_stack(traces)
        if flat_traces is None:
            flat_traces = flat_chain
        else:
            flat_traces = np.append(flat_traces, flat_chain, axis=0)

    labels = list(disp_trace_names)
    # Double all xy labels into x label and y label
    xy_inds = [ind for ind,label in enumerate(labels) if 'xy' in label]
    for ind in reversed(xy_inds):
        label = labels[ind]
        labels[ind] = label.replace('xy', 'y')
        labels.insert(ind, label.replace('xy', 'x'))
    labels = [_axis_label(label) for label in labels]

    if skip_zero_variance:
        # Remove zero-variance variables, because corner.corner will barf
        col_vars = np.var(flat_traces, axis=0)
        variable_cols = np.where(col_vars!=0)[0]
        nonvariable_cols = np.where(col_vars==0)[0]
        flat_traces = flat_traces[:,variable_cols]
        removed_cols = [label for col,label in enumerate(labels)
                        if col in nonvariable_cols]
        labels = [label for col,label in enumerate(labels)
                  if col in variable_cols]
        warn('The following traces had zero variance and will not be '
             'displayed: {}'.format(removed_cols))

    if print_stats:
        chain_len = flat_traces.shape[0] // len(chains)
        chain_slices = [slice(cnum*chain_len, (cnum+1)*chain_len)
                        for cnum in range(len(chains))]
        print('Num chains: {:d} Chain length: {:d}'
              .format(len(chains), chain_len))
        print('Potential Scale Reduction Factors:')
        for col, label in enumerate(labels):
            psr = potential_scale_reduction([flat_traces[slc, col]
                                             for slc in chain_slices])
            print('{} {:0.4f}'.format(label, psr))

    corner(flat_traces, labels=labels, max_n_ticks=3,
           label_kwargs={'fontsize': 'small'}, **kwargs)

    if save:
        pp.savefig('{}_corner.pdf'.format(disp_name))
    else:
        pp.show()
    pp.close(pp.gcf())
    db.close()
