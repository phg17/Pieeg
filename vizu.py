#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:38:24 2020

@author: phg17
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
from scipy import signal
import mne
#from .io import logging
from mne.io.pick import  (channel_type,
                       _VALID_CHANNEL_TYPES, channel_indices_by_type,
                       _DATA_CH_TYPES_SPLIT, _pick_inst, _get_channel_types,
                       _PICK_TYPES_DATA_DICT, _picks_to_idx, pick_info)
from mne.viz.utils import (_draw_proj_checkbox, tight_layout, _check_delayed_ssp,
                    plt_show, _process_times, DraggableColorbar, _setup_cmap,
                    _setup_vmin_vmax, _check_cov, _make_combine_callable,
                    _validate_if_list_of_axes, _triage_rank_sss,
                    _connection_line, _get_color_list, _setup_ax_spines,
                    _setup_plot_projector, _prepare_joint_axes, _check_option,
                    _set_title_multiple_electrodes, _check_time_unit,
                    _plot_masked_image, _trim_ticks)
from mne.viz.topo import _plot_evoked_topo
from mne.utils import (logger, _clean_names, warn, _pl, verbose, _validate_type,
                     _check_if_nan, _check_ch_locs, fill_doc, _is_numeric)
from mne.viz.topomap import (_prepare_topo_plot, _prepare_topomap, plot_topomap,
                      _draw_outlines, _set_contour_locator)
#logging.getLogger('matplotlib').setLevel(logging.WARNING)

#LOGGER = logging.getLogger(__name__.split('.')[0])

PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

def _rgb(x, y, z):
    """Transform x, y, z values into RGB colors."""
    rgb = np.array([x, y, z]).T
    rgb -= rgb.min(0)
    rgb /= np.maximum(rgb.max(0), 1e-16)  # avoid div by zero
    return rgb

def colormap_masked(ncolors=256, knee_index=None, cmap='inferno', alpha=0.3):
    """
    Create a colormap with value below a threshold being greyed out and transparent.
    
    Params
    ------
    ncolors : int
        default to 256
    knee_index : int
        index from which transparency stops
        e.g. knee_index = np.argmin(abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))
    
    Returns
    -------
    cm : LinearSegmentedColormap
        Colormap instance
    """
    cm = plt.cm.get_cmap(cmap)(np.linspace(0, 1, ncolors))
    if knee_index is None:
        # Then map to pvals, as -log(p) between 0 and 3.5, and threshold at 0.05
        knee_index = np.argmin(abs(np.linspace(0., 3.5, ncolors)+np.log10(0.05)))
    
    cm[:knee_index, :] = np.c_[cm[:knee_index, 0], cm[:knee_index, 1], cm[:knee_index, 2], alpha*np.ones((len(cm[:knee_index, 1])))]
    return LinearSegmentedColormap.from_list('my_colormap', cm)

def get_spatial_colors(info):
    "Create set of colours given info (i.e. channel locs) of raw mne object"
    loc3d = np.asarray([el['loc'][:3] for el in info['chs'] if el['kind']==2])
    x, y, z = loc3d.T
    return _rgb(x, y, z)

def plot_filterbank(fbank):
    """
    Plotting a filterbank as created by :func:`pyeeg.preprocess.create_filterbank`
    """
    #plt.plot(w/np.pi,20*np.log10(abs(H)+1e-6))
    signal.freqz(np.stack(fbank)[:, 0, :].T[..., np.newaxis], np.stack(fbank)[:, 1, :].T[..., np.newaxis],
                 plot=lambda w, h: plt.plot(w/np.pi, np.abs(h.T)))
    plt.title('Filter Magnitude Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Amplitude (not in dB)')
    
def plot_filterbank_output(signals, spacing=None, axis=-1):
    """
    Plot output coming out of a filterbank
    Each output of each channel is displayed on top of each other.
    """

    if spacing is None:
        spacing = signals.max()

    for k, filtered in enumerate(signals):
        plt.gca().set_prop_cycle(plt.cycler('color', COLORS[:signals.shape[2]]))
        if axis == -1:
            filtered = filtered.T
        plt.plot(filtered + k*spacing*2)
        
def topomap(arr, info, colorbar=True, ax=None, **kwargs):
    """
    Short-cut to mne topomap...

    Parameters
    ----------
    arr : ndarray (nchan,)
        Array of value to interpolate on a topographic map.
    info : mne.Info instance
        Contains EEG info (channel position for instance)
    
    Returns
    -------
    fig : Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    
    im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, **kwargs)
    if colorbar:
        plt.colorbar(im, ax=ax)
    return fig


def topoplot_array(data, pos, n_topos=1, titles=None):
    """
    Plotting topographic plot.
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    outer_grid = fig.add_gridspec(5, 5, wspace=0.0, hspace=0.25)
    for c in range(n_topos):
        inner_grid = outer_grid[c].subgridspec(1, 1)
        ax = plt.Subplot(fig, inner_grid[0])
        im, _ = mne.viz.plot_topomap(data[:, c], pos, axes=ax, show=False)
        ax.set(title=titles[c])
        fig.add_subplot(ax)

def plot_trf_signi(trf, reject, time_highlight=None, spatial_colors=True, info=None, ax=None, shades=None, **kwargs):
    "Plot trf with significant portions highlighted and with thicker lines"
    trf.plot(ax=ax, **kwargs)

    if spatial_colors:
        assert info is not None, "To use spatial colouring, you must supply raw.info instance"
        colors = get_spatial_colors(info)

    signi_trf = np.ones_like(reject) * np.nan
    list_axes = ax if ax is not None else plt.gcf().axes
    for feat, cax in enumerate(list_axes):
        if shades is None:
            color_shade = 'w' if np.mean(to_rgb(plt.rcParams['axes.facecolor'])) < .5 else [.2, .2, .2]
        else:
            color_shade = shades
        if time_highlight is None:
            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                            where=np.any(reject[:, feat, :], 1),
                            color=color_shade, alpha=0.2)
        else: # fill regions of time of interest
            toi = np.zeros_like(trf.times, dtype=bool)
            for tlims in time_highlight[feat]:
                toi = np.logical_or(toi, np.logical_and(trf.times >= tlims[0], trf.times < tlims[1]))

            cax.fill_between(x=trf.times, y1=cax.get_ylim()[0], y2=cax.get_ylim()[1],
                            where=toi,
                            color=color_shade, alpha=0.2)
        lines = cax.get_lines()
        for k, l in enumerate(lines):
            if spatial_colors:
                l.set_color(np.r_[colors[k], 0.3])
            signi_trf[reject[:, feat, k], feat, k] = l.get_data()[1][reject[:, feat, k]]
        newlines = cax.plot(trf.times, signi_trf[:, feat, :], linewidth=4)
        if spatial_colors:
            for k, l in enumerate(newlines):
                l.set_color(colors[k])
    if ax is None:
        return plt.gcf()

def plots_topogrid(x, y, info, yerr=None, mask=None):
    """
    Display a series of plot arranged in a topographical grid.
    Shaded error bars can be displayed, as well as masking for
    significance portions of data.

    Parameters
    ----------
    x : 1d-array
        Absciss
    y : ndarray
        Data, (ntimes, nchans)
    info : mne.info instance
        info instance containing channel locations
    yerr : ndarry
        Error for shaded areas
    mask : ndarray <bool>
        Boolean array to highlight significant portions of data
        Same shape as y

    Returns
    -------
    fig : figure
    """
    fig = plt.figure(figsize=(12, 10))
    for ax, chan_idx in mne.viz.topo.iter_topography(info,
                                                     fig_facecolor=(36/256, 36/256, 36/256, 0), axis_facecolor='#333333',
                                                     axis_spinecolor='white', fig=fig):
        ax.plot(x, y[:, chan_idx])
        if yerr is not None:
            ax.fill_between(x, y[:, chan_idx] - yerr[:, chan_idx], y[:, chan_idx] + yerr[:, chan_idx],
                            facecolor='C0', edgecolor='C0', linewidth=0, alpha=0.5)
        if mask is not None:
            ax.fill_between(x, ax.get_ylim()[0], ax.get_ylim()[1],
                            where=mask[:, chan_idx].T,
                            facecolor='C2', edgecolor='C2', linewidth=0, alpha=0.5)
        ax.hlines(0, xmin=x[0], xmax=x[-1], linestyle='--', alpha=0.5)
        # Change axes spine color if contains significant portion
        if mask is not None:
            if any(mask[:, chan_idx]):
                for _, v in ax.spines.items():
                    v.set_color('C2')
    return fig


def barplot_annotate_brackets(num1, num2, data, center, height, color='k', yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c=color, linewidth=1.8)

    kwargs_t = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs_t['fontsize'] = fs

    plt.text(*mid, text, color=color, **kwargs_t)
    
def plot_evoked_joint(evoked, times="peaks", title='', picks=None,
                      exclude=None, show=True, ts_args=None,
                      topomap_args=None):
    """Plot evoked data as butterfly plot and add topomaps for time points.
    .. note:: Axes to plot in can be passed by the user through ``ts_args`` or
              ``topomap_args``. In that case both ``ts_args`` and
              ``topomap_args`` axes have to be used. Be aware that when the
              axes are provided, their position may be slightly modified.
    Parameters
    ----------
    evoked : instance of Evoked
        The evoked instance.
    times : float | array of float | "auto" | "peaks"
        The time point(s) to plot. If ``"auto"``, 5 evenly spaced topographies
        between the first and last time instant will be shown. If ``"peaks"``,
        finds time points automatically by checking for 3 local maxima in
        Global Field Power. Defaults to ``"peaks"``.
    title : str | None
        The title. If ``None``, suppress printing channel type title. If an
        empty string, a default title is created. Defaults to ''. If custom
        axes are passed make sure to set ``title=None``, otherwise some of your
        axes may be removed during placement of the title axis.
    %(picks_all)s
    exclude : None | list of str | 'bads'
        Channels names to exclude from being shown. If ``'bads'``, the
        bad channels are excluded. Defaults to ``None``.
    show : bool
        Show figure if ``True``. Defaults to ``True``.
    ts_args : None | dict
        A dict of ``kwargs`` that are forwarded to :meth:`mne.Evoked.plot` to
        style the butterfly plot. If they are not in this dict, the following
        defaults are passed: ``spatial_colors=True``, ``zorder='std'``.
        ``show`` and ``exclude`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.
    topomap_args : None | dict
        A dict of ``kwargs`` that are forwarded to
        :meth:`mne.Evoked.plot_topomap` to style the topomaps.
        If it is not in this dict, ``outlines='skirt'`` will be passed.
        ``show``, ``times``, ``colorbar`` are illegal.
        If ``None``, no customizable arguments will be passed.
        Defaults to ``None``.
    Returns
    -------
    fig : instance of matplotlib.figure.Figure | list
        The figure object containing the plot. If ``evoked`` has multiple
        channel types, a list of figures, one for each channel type, is
        returned.
    Notes
    -----
    .. versionadded:: 0.12.0
    """
    import matplotlib.pyplot as plt

    if ts_args is not None and not isinstance(ts_args, dict):
        raise TypeError('ts_args must be dict or None, got type %s'
                        % (type(ts_args),))
    ts_args = dict() if ts_args is None else ts_args.copy()
    ts_args['time_unit'], _ = _check_time_unit(
        ts_args.get('time_unit', 's'), evoked.times)
    topomap_args = dict() if topomap_args is None else topomap_args.copy()

    got_axes = False
    illegal_args = {"show", 'times', 'exclude'}
    for args in (ts_args, topomap_args):
        if any((x in args for x in illegal_args)):
            raise ValueError("Don't pass any of {} as *_args.".format(
                ", ".join(list(illegal_args))))
    if ("axes" in ts_args) or ("axes" in topomap_args):
        if not (("axes" in ts_args) and ("axes" in topomap_args)):
            raise ValueError("If one of `ts_args` and `topomap_args` contains "
                             "'axes', the other must, too.")
        _validate_if_list_of_axes([ts_args["axes"]], 1)
        n_topomaps = (3 if times is None else len(times)) + 1
        _validate_if_list_of_axes(list(topomap_args["axes"]), n_topomaps)
        got_axes = True

    # channel selection
    # simply create a new evoked object with the desired channel selection
    # Need to deal with proj before picking to avoid bad projections
    proj = topomap_args.get('proj', True)
    proj_ts = ts_args.get('proj', True)
    if proj_ts != proj:
        raise ValueError(
            f'topomap_args["proj"] (default True, got {proj}) must match '
            f'ts_args["proj"] (default True, got {proj_ts})')
    _check_option('topomap_args["proj"]', proj, (True, False, 'reconstruct'))
    evoked = evoked.copy()
    if proj:
        evoked.apply_proj()
        if proj == 'reconstruct':
            evoked._reconstruct_proj()
    topomap_args['proj'] = ts_args['proj'] = False  # don't reapply
    evoked = _pick_inst(evoked, picks, exclude, copy=False)
    info = evoked.info
    ch_types = _get_channel_types(info, unique=True, only_data_chs=True)

    # if multiple sensor types: one plot per channel type, recursive call
    if len(ch_types) > 1:
        if got_axes:
            raise NotImplementedError(
                "Currently, passing axes manually (via `ts_args` or "
                "`topomap_args`) is not supported for multiple channel types.")
        figs = list()
        for this_type in ch_types:  # pick only the corresponding channel type
            ev_ = evoked.copy().pick_channels(
                [info['ch_names'][idx] for idx in range(info['nchan'])
                 if channel_type(info, idx) == this_type])
            if len(_get_channel_types(ev_.info, unique=True)) > 1:
                raise RuntimeError('Possibly infinite loop due to channel '
                                   'selection problem. This should never '
                                   'happen! Please check your channel types.')
            figs.append(
                plot_evoked_joint(
                    ev_, times=times, title=title, show=show, ts_args=ts_args,
                    exclude=list(), topomap_args=topomap_args))
        return figs

    # set up time points to show topomaps for
    times_sec = _process_times(evoked, times, few=True)
    del times
    _, times_ts = _check_time_unit(ts_args['time_unit'], times_sec)

    # prepare axes for topomap
    if not got_axes:
        fig, ts_ax, map_ax, cbar_ax = _prepare_joint_axes(len(times_sec),
                                                          figsize=(8.0, 4.2))
    else:
        ts_ax = ts_args["axes"]
        del ts_args["axes"]
        map_ax = topomap_args["axes"][:-1]
        cbar_ax = topomap_args["axes"][-1]
        del topomap_args["axes"]
        fig = cbar_ax.figure

    # butterfly/time series plot
    # most of this code is about passing defaults on demand
    ts_args_def = dict(picks=None, unit=True, ylim=None, xlim='tight',
                       proj=False, hline=None, units=None, scalings=None,
                       titles=None, gfp=False, window_title=None,
                       spatial_colors=True, zorder='std',
                       sphere=None)
    ts_args_def.update(ts_args)
    _plot_evoked(evoked, axes=ts_ax, show=False, plot_type='butterfly',
                 exclude=[], **ts_args_def)

    # handle title
    # we use a new axis for the title to handle scaling of plots
    old_title = ts_ax.get_title()
    ts_ax.set_title('')
    if title is not None:
        title_ax = plt.subplot(4, 3, 2)
        if title == '':
            title = old_title
        title_ax.text(.5, .5, title, transform=title_ax.transAxes,
                      horizontalalignment='center',
                      verticalalignment='center')
        title_ax.axis('off')

    # topomap
    contours = topomap_args.get('contours', 6)
    ch_type = ch_types.pop()  # set should only contain one element
    # Since the data has all the ch_types, we get the limits from the plot.
    vmin, vmax = ts_ax.get_ylim()
    norm = ch_type == 'grad'
    vmin = 0 if norm else vmin
    vmin, vmax = _setup_vmin_vmax(evoked.data, vmin, vmax, norm)
    if not isinstance(contours, (list, np.ndarray)):
        locator, contours = _set_contour_locator(vmin, vmax, contours)
    else:
        locator = None

    topomap_args_pass = topomap_args.copy()
    topomap_args_pass['outlines'] = topomap_args.get('outlines', 'skirt')
    topomap_args_pass['contours'] = contours
    evoked.plot_topomap(times=times_sec, axes=map_ax, show=False,
                        colorbar=False, **topomap_args_pass)

    if topomap_args.get('colorbar', True):
        from matplotlib import ticker
        cbar = plt.colorbar(map_ax[0].images[0], cax=cbar_ax)
        if isinstance(contours, (list, np.ndarray)):
            cbar.set_ticks(contours)
        else:
            if locator is None:
                locator = ticker.MaxNLocator(nbins=5)
            cbar.locator = locator
        cbar.update_ticks()

    if not got_axes:
        plt.subplots_adjust(left=.1, right=.93, bottom=.14,
                            top=1. if title is not None else 1.2)

    # connection lines
    # draw the connection lines between time series and topoplots
    lines = [_connection_line(timepoint, fig, ts_ax, map_ax_)
             for timepoint, map_ax_ in zip(times_ts, map_ax)]
    for line in lines:
        fig.lines.append(line)

    # mark times in time series plot
    for timepoint in times_ts:
        ts_ax.axvline(timepoint, color='grey', linestyle='-',
                      linewidth=1.5, alpha=.66, zorder=0)

    # show and return it
    plt_show(show)
    return fig

def _plot_evoked(evoked, picks, exclude, unit, show, ylim, proj, xlim, hline,
                 units, scalings, titles, axes, plot_type, cmap=None,
                 gfp=False, window_title=None, spatial_colors=False,
                 selectable=True, zorder='unsorted',
                 noise_cov=None, colorbar=True, mask=None, mask_style=None,
                 mask_cmap=None, mask_alpha=.25, time_unit='s',
                 show_names=False, group_by=None, sphere=None):
    """Aux function for plot_evoked and plot_evoked_image (cf. docstrings).
    Extra param is:
    plot_type : str, value ('butterfly' | 'image')
        The type of graph to plot: 'butterfly' plots each channel as a line
        (x axis: time, y axis: amplitude). 'image' plots a 2D image where
        color depicts the amplitude of each channel at a given time point
        (x axis: time, y axis: channel). In 'image' mode, the plot is not
        interactive.
    """
    import matplotlib.pyplot as plt

    # For evoked.plot_image ...
    # First input checks for group_by and axes if any of them is not None.
    # Either both must be dicts, or neither.
    # If the former, the two dicts provide picks and axes to plot them to.
    # Then, we call this function recursively for each entry in `group_by`.
    if plot_type == "image" and isinstance(group_by, dict):
        if axes is None:
            axes = dict()
            for sel in group_by:
                plt.figure()
                axes[sel] = plt.axes()
        if not isinstance(axes, dict):
            raise ValueError("If `group_by` is a dict, `axes` must be "
                             "a dict of axes or None.")
        _validate_if_list_of_axes(list(axes.values()))
        remove_xlabels = any([_is_last_row(ax) for ax in axes.values()])
        for sel in group_by:  # ... we loop over selections
            if sel not in axes:
                raise ValueError(sel + " present in `group_by`, but not "
                                 "found in `axes`")
            ax = axes[sel]
            # the unwieldy dict comp below defaults the title to the sel
            titles = ({channel_type(evoked.info, idx): sel
                       for idx in group_by[sel]} if titles is None else titles)
            _plot_evoked(evoked, group_by[sel], exclude, unit, show, ylim,
                         proj, xlim, hline, units, scalings, titles,
                         ax, plot_type, cmap=cmap, gfp=gfp,
                         window_title=window_title,
                         selectable=selectable, noise_cov=noise_cov,
                         colorbar=colorbar, mask=mask,
                         mask_style=mask_style, mask_cmap=mask_cmap,
                         mask_alpha=mask_alpha, time_unit=time_unit,
                         show_names=show_names,
                         sphere=sphere)
            if remove_xlabels and not _is_last_row(ax):
                ax.set_xticklabels([])
                ax.set_xlabel("")
        ims = [ax.images[0] for ax in axes.values()]
        clims = np.array([im.get_clim() for im in ims])
        min, max = clims.min(), clims.max()
        for im in ims:
            im.set_clim(min, max)
        figs = [ax.get_figure() for ax in axes.values()]
        if len(set(figs)) == 1:
            return figs[0]
        else:
            return figs
    elif isinstance(axes, dict):
        raise ValueError("If `group_by` is not a dict, "
                         "`axes` must not be a dict either.")

    time_unit, times = _check_time_unit(time_unit, evoked.times)
    evoked = evoked.copy()  # we modify info
    info = evoked.info
    if axes is not None and proj == 'interactive':
        raise RuntimeError('Currently only single axis figures are supported'
                           ' for interactive SSP selection.')
    if isinstance(gfp, str) and gfp != 'only':
        raise ValueError('gfp must be boolean or "only". Got %s' % gfp)

    scalings = _handle_default('scalings', scalings)
    titles = _handle_default('titles', titles)
    units = _handle_default('units', units)

    picks = _picks_to_idx(info, picks, none='all', exclude=())
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")

    bad_ch_idx = [info['ch_names'].index(ch) for ch in info['bads']
                  if ch in info['ch_names']]
    if len(exclude) > 0:
        if isinstance(exclude, str) and exclude == 'bads':
            exclude = bad_ch_idx
        elif (isinstance(exclude, list) and
              all(isinstance(ch, str) for ch in exclude)):
            exclude = [info['ch_names'].index(ch) for ch in exclude]
        else:
            raise ValueError(
                'exclude has to be a list of channel names or "bads"')

        picks = np.array([pick for pick in picks if pick not in exclude])

    types = np.array(_get_channel_types(info, picks), str)
    ch_types_used = list()
    for this_type in _VALID_CHANNEL_TYPES:
        if this_type in types:
            ch_types_used.append(this_type)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(len(ch_types_used), 1)
        fig.subplots_adjust(left=0.125, bottom=0.1, right=0.975, top=0.92,
                            hspace=0.63)
        if isinstance(axes, plt.Axes):
            axes = [axes]
        fig.set_size_inches(6.4, 2 + len(axes))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = list(axes)

    if fig is None:
        fig = axes[0].get_figure()

    if window_title is not None:
        _set_window_title(fig, window_title)

    if len(axes) != len(ch_types_used):
        raise ValueError('Number of axes (%g) must match number of channel '
                         'types (%d: %s)' % (len(axes), len(ch_types_used),
                                             sorted(ch_types_used)))
    _check_option('proj', proj, (True, False, 'interactive', 'reconstruct'))
    noise_cov = _check_cov(noise_cov, info)
    if proj == 'reconstruct' and noise_cov is not None:
        raise ValueError('Cannot use proj="reconstruct" when noise_cov is not '
                         'None')
    projector, whitened_ch_names = _setup_plot_projector(
        info, noise_cov, proj=proj is True, nave=evoked.nave)
    if len(whitened_ch_names) > 0:
        unit = False
    if projector is not None:
        evoked.data[:] = np.dot(projector, evoked.data)
    if proj == 'reconstruct':
        evoked = evoked._reconstruct_proj()

    if plot_type == 'butterfly':
        _plot_lines(evoked.data, info, picks, fig, axes, spatial_colors, unit,
                    units, scalings, hline, gfp, types, zorder, xlim, ylim,
                    times, bad_ch_idx, titles, ch_types_used, selectable,
                    False, line_alpha=1., nave=evoked.nave,
                    time_unit=time_unit, sphere=sphere)
        plt.setp(axes, xlabel='Time (%s)' % time_unit)

    elif plot_type == 'image':
        for ai, (ax, this_type) in enumerate(zip(axes, ch_types_used)):
            use_nave = evoked.nave if ai == 0 else None
            this_picks = list(picks[types == this_type])
            _plot_image(evoked.data, ax, this_type, this_picks, cmap, unit,
                        units, scalings, times, xlim, ylim, titles,
                        colorbar=colorbar, mask=mask, mask_style=mask_style,
                        mask_cmap=mask_cmap, mask_alpha=mask_alpha,
                        nave=use_nave, time_unit=time_unit,
                        show_names=show_names, ch_names=evoked.ch_names)
    if proj == 'interactive':
        _check_delayed_ssp(evoked)
        params = dict(evoked=evoked, fig=fig, projs=info['projs'], axes=axes,
                      types=types, units=units, scalings=scalings, unit=unit,
                      ch_types_used=ch_types_used, picks=picks,
                      plot_update_proj_callback=_plot_update_evoked,
                      plot_type=plot_type)
        _draw_proj_checkbox(None, params)

    plt.setp(fig.axes[:len(ch_types_used) - 1], xlabel='')
    fig.canvas.draw()  # for axes plots update axes.
    plt_show(show)
    return fig