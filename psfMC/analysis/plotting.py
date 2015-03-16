# coding=utf-8
from __future__ import division
import pymc
import matplotlib.pyplot as pp
import numpy as np
from matplotlib import patheffects
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .statistics import num_effective_samples, potential_scale_reduction, \
    max_posterior_sample, chain_autocorr
from ..models import multicomponent_model
from ..ModelComponents.Sersic import Sersic
from scipy.special import gamma

_labels = {'Sky_adu': 'Sky (adu)',
           'PSF_mag': 'Point source mag',
           'Sersic_mag': u'Sérsic profile mag',
           'Sersic_index': u'Sérsic index $n$',
           'Sersic_reff': u'Sérsic effective radius $r_e$ (pixels)',
           'Sersic_reff_b': u'Sérsic minor axis (pixels)',
           'Sersic_angle': u'Sérsic profile position angle',
           'deviance': 'Deviance (-2*log(P))',
           'magdiff': '$m_{host} - m_{quasar}$',
           'centerdist': 'Host-quasar position difference (pixels)',
           'axisratio': 'Host axis ratio $b/a$',
           'sbeff': 'Characteristic surface brightness $\mu_e$ (mag '
                    'pixel$^{-1}$)'}

_text_stroke = patheffects.withStroke(linewidth=3, foreground='w')


def _axis_label(trace_name):
    """
    Returns an axis label for the specified trace name
    """
    if trace_name in _labels.keys():
        return _labels[trace_name]
    elif '_' in trace_name:
        index, label_key = trace_name.split('_', 1)
        return u'({}) {}'.format(index, _labels.get(label_key, label_key))
    else:
        return trace_name


def _sbeff(mag, reff, reff_b, index, px_scale=None):
    kappa = Sersic.kappa(index)
    to_magarc = 2.5*np.log10(px_scale**2) if px_scale is not None else 0
    return mag + 2.5*np.log10(2 * np.pi * reff * reff_b * index *
                              np.exp(kappa) / kappa**(2*index) *
                              gamma(2*index)) + to_magarc


def _get_prior(key, model, values):
    """
    :param key: trace name
    :param model: psfMC model
    :param values: values to calculate probabilities for
    """
    prior_logp = model.get_node(key).__dict__['_logp'].fun
    prior_args = dict(model.get_node(key).__dict__['_logp'].arguments)
    del prior_args['value']

    return np.exp([prior_logp(value=val, **prior_args) for val in values])


def _select_chains(db, chains):
    """
    Select database chain indexes. Allows for negative indexes (i.e. last chain,
    second-to-last chain, etc.).
    """
    if chains is None:
        return range(db.chains)
    else:
        return np.arange(db.chains)[chains]


def get_trace(key, db, chain=1):
    trace = None
    # TODO: bother keeping these? Need to make them generic
    try:
        if key == 'magdiff':
            trace = (db.trace('2_Sersic_mag', chain)[:] -
                     db.trace('1_PSF_mag', chain)[:])
        elif key == 'centerdist':
            cdiff = (db.trace('1_PSF_xy', chain)[:] -
                     db.trace('2_Sersic_xy', chain)[:])
            trace = np.sqrt(np.sum(cdiff**2, axis=1))
        elif key == 'axisratio':
            trace = (db.trace('2_Sersic_reff_b', chain)[:] /
                     db.trace('2_Sersic_reff', chain)[:])
        elif key == 'sbeff':
            trace = _sbeff(db.trace('2_Sersic_mag', chain)[:],
                           db.trace('2_Sersic_reff', chain)[:],
                           db.trace('2_Sersic_reff_b', chain)[:],
                           db.trace('2_Sersic_index', chain)[:])
        else:
            trace = db.trace(key, chain)[:]
    except KeyError, err:
        names = db.trace_names[chain]
        err.message = "Trace '{}' not found while plotting {}. Available " \
            "traces are 'magdiff', 'centerdist', 'axisratio', 'sbeff' or: {}"\
            .format(err, key, names)
        raise err

    # For 1D traces (all except XY coordinates), expand to 2D array
    if len(trace.shape) == 1:
        trace = np.expand_dims(trace, 1)
    return trace


def _load_db_and_model(db, model):
    # TODO: Keep this stuff that requires path checking?
    import os
    disp_name, ext = os.path.splitext(os.path.basename(db))
    db_module = getattr(pymc.database, ext[1:])
    db = db_module.load(db)

    if model is not None:
        try:
            model_dir = os.path.dirname(model)
            if model_dir == '':
                model_dir = '.'
            prev_dir = os.getcwd()
            os.chdir(model_dir)
            model = multicomponent_model(os.path.basename(model))
            os.chdir(prev_dir)
        except IOError, err:
            print 'Unable to find model file {}. Priors will not be plotted.'\
                .format(model)
            model = None
    return disp_name, db, model


def plot_trace(key, db, chains=None, save=False):
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
        trace = get_trace(key, db, chain)

        for col in xrange(trace.shape[1]):
            ax_trace.plot(np.arange(trace.shape[0]), trace[:, col])
            ax_hist.hist(trace[:, col], bins=20, histtype='step',
                         orientation='horizontal')
            ml_val = trace[ml_model, col]
            ax_hist.axhline(ml_val, color='Orange', lw=2)

    ax_trace.set_xlabel('Sample')
    ax_trace.set_ylabel(_axis_label(key))
    pp.gcf().suptitle(disp_name)

    if save:
        pp.savefig('_'.join([disp_name, key, 'trace.pdf']))
    else:
        pp.show()
    pp.close(pp.gcf())
    db.close()


def plot_hist(key, db, model=None, chains=None, save=False):
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
        trace = get_trace(key, db, chain)
        ml_model, ml_chain = max_posterior_sample(db, [chain])

        all_traces.append(trace)
        for col in xrange(trace.shape[1]):
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
           for col in xrange(all_traces[0].shape[1])]
    neff = [num_effective_samples([trace[:, col] for trace in all_traces])
            for col in xrange(all_traces[0].shape[1])]

    fmt = '\n'.join(['$\widehat{{R}}$ = {:0.3f}']*len(psr) +
                    ['$n_{{eff}}$ = {:0.1f}']*len(neff))

    ax_hist.text(0.95, 0.95, fmt.format(*(psr+neff)),
                 ha='right', va='top',  # path_effects=[_text_stroke],
                 transform=ax_hist.transAxes)

    ax_hist.set_xlabel(_axis_label(key))
    ax_hist.set_ylabel('Number of Samples')
    fig_hist.suptitle(disp_name)

    ax_autocorr.set_xlim(0, maxlag+10)
    ax_autocorr.axhline(0.0, color='black')
    # ax_autocorr.axhline(-acorr_signif, color='gray', ls='dashed')
    # ax_autocorr.axhline(acorr_signif, color='gray', ls='dashed')
    autocorr_labels += ['$n_{{eff,tot}}$ = {:0.1f}'.format(neff_autocorr)]
    ax_autocorr.text(0.95, 0.95, '\n'.join(autocorr_labels),
                     va='top', ha='right',  # path_effects=[_text_stroke],
                     transform=ax_autocorr.transAxes)
    ax_autocorr.set_xlabel('Lag Length (Samples)')
    ax_autocorr.set_ylabel('Autocorrelation (Normalized)')
    fig_acorr.suptitle(disp_name)

    # FIXME: TEMP PRIOR SHIT
    if model is not None:
        min_x, max_x = np.min(all_traces), np.max(all_traces)
        mid = 0.5*(min_x + max_x)
        ptp = (max_x - min_x) * 1.1  # 5% higher/lower than min/max
        prior_x = np.linspace(mid-ptp*0.5, mid+ptp*0.5, 100)
        prior_y = _get_prior(key, model, prior_x)

        prior_xform = blended_transform_factory(ax_hist.transData,
                                                ax_hist.transAxes)
        ax_hist.plot(prior_x, prior_y, lw=3, color='black',
                     transform=prior_xform)

    if save:
        fig_hist.savefig('_'.join([disp_name, key, 'hist.pdf']))
        fig_acorr.savefig('_'.join([disp_name, key, 'acorr.pdf']))
        fmt = '{} mu: {:0.5g} sig: {:0.3g} R: {} ne: {} nea: {}'
        all_traces = np.array(all_traces)
        mean = np.mean(all_traces)
        std = np.std(all_traces)
        print fmt.format(key, mean, std, psr, neff, neff_autocorr)
    else:
        pp.show()
    pp.close(fig_hist)
    pp.close(fig_acorr)
    db.close()
