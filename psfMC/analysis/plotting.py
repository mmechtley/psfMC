# coding=utf-8
from __future__ import division, print_function
import os
from warnings import warn

import matplotlib.pyplot as pp
import numpy as np
from matplotlib import patheffects
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from corner import corner
from emcee import autocorr
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

from .statistics import chain_autocorr
from ..database import load_database
from ..models import MultiComponentModel
from ..ModelComponents.Sersic import Sersic


_labels = {'lnprobability': 'Model posterior log-probability',
           'x': '{} x (pix)',
           'y': '{} y (pix)',
           'xy': '{} x,y (pix)',
           'adu': '{} (adu)',
           'mag': '{} mag',
           'index': '{} index $n$',
           'reff': '{} $R_e a$ (pix)',
           'reff_b': '{} $R_e b$ (pix)',
           'angle': '{} PA (deg)',
           'PSF_Index': 'PSF index',
           'axisratio': '{} axis ratio $b/a$',
           'sbeff': '{} $\mu_e$ (mag arcsec$^2$)',
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


def _get_trace(trace_name, db, model=None):
    """
    Get the trace array for the specified parameter key. Also handles lookup and
    calculation of the special keys magdiff, centerdist, axisratio, and sbeff.
    e.g. 1_PSF_2_Sersic_centerdist or 2_Sersic_sbeff
    :param trace_name: Name of the model parameter to get the trace for
    :param db: astropy Table object to get trace from
    :return: Trace values as NxD array, where N is the number of samples and D
        is the number of dimensions of the parameter (2 for xy, 1 for others)
    """
    try:
        name_comps = trace_name.split('_')
        if 'magdiff' in name_comps:
            key_mag1 = '_'.join(name_comps[0:2] + ['mag'])
            key_mag2 = '_'.join(name_comps[2:4] + ['mag'])
            trace = db[key_mag1] - db[key_mag2]
        elif 'centerdist' in name_comps:
            key_xy1 = '_'.join(name_comps[0:2] + ['xy'])
            key_xy2 = '_'.join(name_comps[2:4] + ['xy'])
            cdiff = db[key_xy1] - db[key_xy2]
            trace = np.sqrt(np.sum(cdiff**2, axis=1))
        elif 'axisratio' in name_comps:
            key_prefix = '_'.join(name_comps[0:2] + [''])
            trace = db[key_prefix + 'reff_b'] / db[key_prefix + 'reff']
        elif 'sbeff' in name_comps:
            key_prefix = '_'.join(name_comps[0:2] + [''])
            trace = Sersic.mag_to_flux(db[key_prefix + 'mag'], 0)
            trace = Sersic.sb_eff(trace,
                                  db[key_prefix + 'index'],
                                  db[key_prefix + 'reff'],
                                  db[key_prefix + 'reff_b'])
            if model is not None:
                wcs = WCS(model.obs_header)
                px_area = proj_plane_pixel_area(wcs)
                px_area *= 3600**2  # to sq arcsec
                trace /= px_area
            trace = -2.5 * np.log10(trace)
        else:
            trace = db[trace_name]
    except KeyError as err:
        names = db.colnames
        err.message = 'Unable to find trace {} while plotting {}. Available '\
            'traces are {} or magdiff, centerdist, axisratio, sbeff'\
            .format(err, trace_name, names)
        raise err

    # For 1D traces (all except XY coordinates), expand to Nx1 2D array
    if len(trace.shape) == 1:
        trace = np.expand_dims(trace, 1)
    return trace


def _load_db_and_model(db_file, model_file):
    """
    Open database file and model file, returning display name (database name
    without extension), database Table object, and Model object.
    """
    # TODO: Allow already-opened database and model, make public
    disp_name, ext = os.path.splitext(os.path.basename(db_file))
    db = load_database(db_file)

    model = None
    if model_file is not None:
        try:
            model = MultiComponentModel(model_file)
        except IOError:
            print('Unable to find model file {}. Priors will not be plotted.'
                  .format(model_file))
    return disp_name, db, model


def plot_trace(trace_name, db, model=None, save=False):
    """
    Plot a trace value vs. time (or sample number). Useful for assessing if the
    sampler has reached a stable distribution.
    :param trace_name: Name of traced quantity, including all with priors, as
        well as derived quantities magdiff, centerdist, sbeff, and axisratio.
    :param db: Filename of trace database
    :param model: (Optional) filename of model file (for sbeff in mag arcsec^2)
    :param save: If True, plots will not be displayed but will be saved to disk
        in pdf format.
    """
    disp_name, db, model = _load_db_and_model(db, model)

    fig = pp.figure()
    ax_trace = pp.subplot(111)

    divider = make_axes_locatable(ax_trace)
    ax_hist = divider.append_axes('right', size=1.2, pad=0.1, sharey=ax_trace)

    ticker = MaxNLocator(nbins=3, integer=True)
    ax_hist.get_xaxis().set_major_locator(ticker)
    pp.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.get_xaxis().tick_top()

    best_row = np.argmax(db['lnprobability'])
    trace = _get_trace(trace_name, db, model=model)
    n_walkers = db['walker'].max() + 1
    n_samples = trace.shape[0] // n_walkers

    for col in range(trace.shape[1]):
        for walker in range(n_walkers):
            walker_trace = trace[:, col][db['walker'] == walker]
            ax_trace.plot(np.arange(n_samples), walker_trace,
                          color='black', alpha=0.3)
        ax_hist.hist(trace[:, col], bins=20, histtype='step',
                     orientation='horizontal')
        ml_val = trace[best_row, col]
        ax_hist.axhline(ml_val, color='Orange', lw=2)

    ax_trace.set_xlabel('Sample')
    ax_trace.set_ylabel(_axis_label(trace_name))
    fig.suptitle(disp_name)

    if save:
        pp.savefig('_'.join([disp_name, trace_name, 'trace.pdf']))
    else:
        pp.show()
    pp.close(fig)


def plot_hist(trace_name, db, model=None, save=False):
    """
    Plot histogram for the given traced quantity. multi-d quantities (xy
    positions) will be plotted as multiple histograms on the same axes. All
    walkers will be combined into a single histogram

    :param trace_name: Name of traced quantity, including all with priors, as
        well as derived quantities: magdiff, centerdist, sbeff, and axisratio.
    :param db: Filename of psfMC database
    :param model: Filename of psfMC model. If None, priors will not be plotted.
    :param save: If True, plots will not be displayed but will be saved to disk
        in pdf format.
    """
    disp_name, db, model = _load_db_and_model(db, model)

    fig_hist = pp.figure()
    ax_hist = fig_hist.add_subplot(111)

    trace = _get_trace(trace_name, db, model=model)
    best_row = np.argmax(db['lnprobability'])

    for col in range(trace.shape[1]):
        ax_hist.hist(trace[:, col], bins=20, histtype='step', lw=2)
        ml_val = trace[best_row, col]
        ax_hist.axvline(ml_val, lw=2, ls='dashed')

    fig_hist.suptitle(disp_name)
    ax_hist.set_xlabel(_axis_label(trace_name))
    ax_hist.set_ylabel('Number of Samples')

    # Display the prior for this traced variable, if it has a prior
    if model is not None:
        prior = model.get_distribution(trace_name)
    else:
        prior = None
    if prior is not None:
        min_xs, max_xs = prior.rv_frozen.interval(0.99)
        min_xs = np.atleast_1d(min_xs)
        max_xs = np.atleast_1d(max_xs)
        min_xs -= 0.01*(max_xs - min_xs)  # 1% higher/lower than min/max
        max_xs += 0.01*(max_xs - min_xs)

        prior_x = np.column_stack([np.linspace(dim_min, dim_max, 100)
                                   for dim_min, dim_max in zip(min_xs, max_xs)])

        # display prior with normalized (axes) y coordinate
        prior_xform = blended_transform_factory(ax_hist.transData,
                                                ax_hist.transAxes)
        ax_hist.plot(prior_x, np.exp(prior.logp(prior_x)),
                     lw=1, color='black', zorder=-1,
                     transform=prior_xform)

    if save:
        fig_hist.savefig('_'.join([disp_name, trace_name, 'hist.pdf']))
    else:
        pp.show()
    pp.close(fig_hist)


def plot_autocorr(trace_name, db, save=False):
    """
    Plot autocorrelation diagrams for a given traced quantity. For ensemble
    (multi-walker) database data, the mean of all walkers for the given traced
    quantity is used to estimate autocorrelation (same as emcee)

    :param trace_name: Name of traced quantity, including all with priors, as
        well as derived quantities: magdiff, centerdist, sbeff, and axisratio.
    :param db: Filename of psfMC database
    :param save: If True, plots will not be displayed but will be saved to disk
        in pdf format.
    """
    disp_name, db, model = _load_db_and_model(db, None)

    fig_acorr = pp.figure()
    ax_acorr = fig_acorr.add_subplot(111)

    autocorr_labels = []
    maxlag = 0

    trace = _get_trace(trace_name, db)
    n_walkers = db['walker'].max() + 1
    n_samples = trace.shape[0] // n_walkers

    for col in range(trace.shape[1]):
        trace_walkers = trace[:, col].reshape((n_samples, n_walkers), order='F')

        # TODO: replace with emcee autocorr?
        lags, corr, eff_samples, chain_maxlag = chain_autocorr(trace_walkers)
        emcee_acorr = autocorr.function(trace_walkers)

        maxlag = np.max([maxlag, chain_maxlag])

        for walk in range(n_walkers):
            ax_acorr.plot(np.arange(n_samples), emcee_acorr[:, walk],
                          marker=None, ls='solid', lw=1, color='black',
                          alpha=0.3, drawstyle='steps-mid')
        ax_acorr.plot(lags, corr, marker=None, ls='solid', lw=2,
                      drawstyle='steps-mid')
        ax_acorr.axvline(chain_maxlag)

        autocorr_labels += ['$n_{{eff}}$ = {:0.1f}'.format(eff_samples)]

    fig_acorr.suptitle(disp_name)
    ax_acorr.set_xlim(0, maxlag + 10)
    ax_acorr.axhline(0.0, color='black')
    ax_acorr.set_xlabel('Lag Length (Samples)')
    ax_acorr.set_ylabel('Autocorrelation (Normalized)')

    ax_acorr.text(0.95, 0.95, '\n'.join(autocorr_labels),
                  va='top', ha='right',
                  transform=ax_acorr.transAxes)

    if save:
        fig_acorr.savefig('_'.join([disp_name, trace_name, 'acorr.pdf']))
    else:
        pp.show()
    pp.close(fig_acorr)


def corner_plot(database, disp_parameters=None, save=False,
                skip_zero_variance=True, **kwargs):
    """


    :param database: Filename of psfMC database
    :param disp_parameters: List/Tuple of stochastic variable (column) names
        from the database to plot
    :param save: Save as pdf instead of displaying to screen
    :param skip_zero_variance: Remove/skip traced columns with 0 variance. If
        False, corner will raise an error when trying to plot them.
    :param kwargs: Additional arguments passed to corner.corner
    """
    disp_name, db, model = _load_db_and_model(database, None)
    # find available traces
    available_col_names = db.colnames

    # If no traces were specified, remove probability and walker
    if disp_parameters is None:
        display_col_names = list(available_col_names)
        display_col_names.remove('lnprobability')
        display_col_names.remove('walker')
    # If we specified traces, display those ones
    else:
        missing_traces = set(disp_parameters) - set(available_col_names)
        if missing_traces != set():
            raise ValueError('Unable to find trace(s) named: {}'.format(
                missing_traces))
        display_col_names = list(disp_parameters)

    # Assemble traces as giant data array (samples x parameters), as expected by
    # corner.corner()
    traces = [_get_trace(trace_name, db) for trace_name in display_col_names]
    flat_traces = np.column_stack(traces)

    labels = list(display_col_names)
    # Double all xy labels into x label and y label
    xy_inds = [ind for ind, label in enumerate(labels) if 'xy' in label]
    for ind in reversed(xy_inds):
        label = labels[ind]
        labels[ind] = label.replace('xy', 'y')
        labels.insert(ind, label.replace('xy', 'x'))
    labels = [_axis_label(label) for label in labels]

    if skip_zero_variance:
        # Remove zero-variance variables, because corner.corner will barf
        col_vars = np.var(flat_traces, axis=0)
        variable_cols = np.where(col_vars != 0)[0]
        nonvariable_cols = np.where(col_vars == 0)[0]
        flat_traces = flat_traces[:, variable_cols]
        removed_cols = [label for col, label in enumerate(labels)
                        if col in nonvariable_cols]
        labels = [label for col, label in enumerate(labels)
                  if col in variable_cols]
        if len(removed_cols) > 0:
            warn('The following traces had zero variance and will not be '
                 'displayed: {}'.format(removed_cols))

    fig = corner(flat_traces, labels=labels, max_n_ticks=3,
                 label_kwargs={'fontsize': 'small'}, **kwargs)

    if save:
        pp.savefig('{}_corner.pdf'.format(disp_name))
    else:
        pp.show()
    pp.close(fig)
