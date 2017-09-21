import obspy
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
from obspy.signal.util import next_pow_2
import harmonics
import matplotlib.patches as patches
import matplotlib as mpl
from fit_poly import pick_peaks

mpl.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['agg.path.chunksize'] = 10000


def _pick_longperiod(t, f, s, fmin=0.01, fmax=0.1):
    # pick time windows with high long-period noise level
    slope = []
    cross = []
    for i in range(0, len(t)):
        # Reduce to periods between 10 and 100s
        x = 1. / f[f > fmin]
        y = s[f > fmin, i]
        y = y[x < 1./fmax]
        x = x[x < 1./fmax]

        a, b = np.polyfit(x, y, deg=1)
        slope.append(a)
        cross.append(b)

    times = []
    level = []

    for time, crossi in zip(t, cross):
        if crossi > np.median(cross) + 5:
            times.append(time)
            level.append(crossi)

    return times, level


def write_picks(path, times, picks, peakvals, stats):
    fnam_out = os.path.join(path, 'picks_%s_%s.txt' %
                            (stats.station, stats.channel))

    with open(fnam_out, 'a') as fid:
        for t, p, peak in zip(times, picks, peakvals):
            t_out = stats.starttime + t
            fid.write('%s, %f, %f\n' % (t_out, p, peak))


def time_of_day(time):
    return time.hour*3600 + time.minute*60 + time.second


def _integrate_displacement(t, f, s, peaks_periods, width):
        disp_int = np.zeros(len(t))
        for i in range(0, len(t)):
            fmin_integrate = 1. / (peaks_periods[i] + width/2)
            fmax_integrate = 1. / (peaks_periods[i] - width/2)
            bol_peak = np.array((f > fmin_integrate,
                                 f < fmax_integrate)).all(axis=0)
            df = f[2] - f[1]
            s_i = s[:, i]
            disp_int[i] = np.sum(np.sqrt(s_i[bol_peak])) * df
        return disp_int


def calc_spec(fnam_smgr, fnam_aux=None, fmin=1e-2, fmax=10, 
              vmin=-180, vmax=-80, winlen=300,
              pick_harmonics=False, fmin_pick=0.4, fmax_pick=1.2,
              plot_highfreq=True, pick_peak=True, 
              fmin_plot=None, fmax_plot=None,
              s_threshold=-140, path_out='.', nharms=4, sigma_min=1e-3,
              p_peak_min=5, dpi=100, cat=None):

    st = obspy.read(fnam_smgr)
    if len(st) > 1:
        raise ValueError('Only one trace per seismogram file, please!')
    else:
        tr = st[0]
    
    if tr.stats.sampling_rate > 25:
        tr.decimate(int(tr.stats.sampling_rate / 25))

    # Workaround for the stations starting one sample early
    tr.stats.starttime += 1

    # Check whether spectrogram image already exists
    fnam_pic = '%s.%s_%04d_%02d_%02d_%02d.png' % (tr.stats.station,
                                                  tr.stats.channel,
                                                  tr.stats.starttime.year,
                                                  tr.stats.starttime.month,
                                                  tr.stats.starttime.day,
                                                  tr.stats.starttime.hour)

    s, f, t = specgram(tr.data, Fs=tr.stats.sampling_rate,
                       NFFT=winlen*tr.stats.sampling_rate,
                       pad_to=next_pow_2(winlen*tr.stats.sampling_rate) * 2,
                       noverlap=winlen*tr.stats.sampling_rate*0.5)

    t += time_of_day(tr.stats.starttime)

    slog = 10 * np.log10(s)
    f += 1e-10

    fig = plt.figure(figsize=(16, 8))

    # Seismogram axis
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax1b = ax1.twinx()

    if plot_highfreq:
        # Spectrogram axis (f>1Hz)
        ax2a = fig.add_axes([0.1, 0.45, 0.7, 0.3], sharex=ax1)
        # Spectrogram axis (p>1s)
        ax2b = fig.add_axes([0.1, 0.1, 0.7, 0.35], sharex=ax1)
    else:
        # Spectrogram axis (p>1s)
        ax2b = fig.add_axes([0.1, 0.1, 0.7, 0.65], sharex=ax1)

    # Colorbar axis
    ax3 = fig.add_axes([0.83, 0.8, 0.03, 0.15])

    if plot_highfreq:
        # Mean PSD axis (f>1Hz)
        ax4a = fig.add_axes([0.8, 0.45, 0.1, 0.3], sharey=ax2a)
        # Mean PSD axis (p>1s)
        ax4b = fig.add_axes([0.8, 0.1, 0.1, 0.35], sharey=ax2b)
    else:
        # Mean PSD axis (p>1s)
        ax4b = fig.add_axes([0.8, 0.1, 0.1, 0.65], sharey=ax2b)

    # Axis with seismogram
    tr_filt = tr.copy()

    # Filter, if desired
    if fmin_plot:
        tr_filt.filter('highpass', freq=fmin_plot)
    if fmax_plot:
        tr_filt.filter('lowpass', freq=fmax_plot)

    ax1.plot(tr.times() + time_of_day(tr.stats.starttime), 
             tr_filt.data, 
             'k', linewidth=0.5)
    ymaxmin = np.percentile(abs(tr_filt.data), q=99.99) * 1.1
    ax1.set_ylim(-ymaxmin, ymaxmin)

    if tr.stats.channel == 'BDH':
        ax1.set_ylabel('Pressure / Pa')
    else:
        ax1.set_ylabel('Displacement / m')
    ax1.grid('on')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # If we ahave an auxilliary dataset, plot it in here.
    if fnam_aux:
        tr_aux = obspy.read(fnam_aux)[0]
        tr_aux.trim(tr.stats.starttime, tr.stats.endtime)
        ax1b.plot(tr_aux.times() + time_of_day(tr_aux.stats.starttime), 
                 tr_aux.data, 
                 'r', linewidth=0.5)

        ax1b.set_ylim(min(tr_aux.data), max(tr_aux.data))
        ax1b.tick_params('y', colors='r')


    if plot_highfreq:
        # Axis with high frequency part of spectrogram
        ax2a.pcolormesh(t, f, slog,
                        vmin=vmin, vmax=vmax, cmap='plasma')
        ax2a.set_ylim(1, f[-1])
        ax2a.set_ylabel('frequency / Hz')
        plt.setp(ax2a.get_xticklabels(), visible=False)
        ax2a.grid('on')

        if (pick_harmonics):
            times, freqs, peakvals, fmax_used = \
                harmonics.pick_spectrogram(f, t, slog,
                                           fwin=[fmin_pick,
                                                 fmax_pick],
                                           sigma_min=sigma_min,
                                           p_peak_min=p_peak_min,
                                           winlen=winlen,
                                           nharms=nharms,
                                           s_threshold=s_threshold)
            for time, freq in zip(times, freqs):
                posx = time - winlen/4
                posy = fmax_pick
                patchi = patches.Rectangle((posx, posy),
                                           width=winlen/2, height=fmax_pick * 2,
                                           alpha=0.1,
                                           color=(0.8, 0, 0), edgecolor=None)
                ax2a.add_patch(patchi)

                ax2a.plot(time, freq, 'ko', alpha=0.2)

            ax2a.hlines((fmin_pick, fmax_pick), 0, t[-1],
                        colors='k', linestyles='dashed')
            # Write Picks to file
            write_picks(os.path.join(path_out, 'Picks'),
                        times, freqs, peakvals, tr.stats)




    # Axis with low frequency part of spectrogram
    ax2b.pcolormesh(t, np.log10(1./f) + 1., slog,
                    vmin=vmin, vmax=vmax, cmap='plasma')
    ax2b.set_ylim(np.log10(1./fmin), 1)
    ax2b.set_xlabel('time of day')
    ax2b.set_ylabel('period / seconds')
    ax2b.set_xlim(0, t[-1])

    ax2b.set_yticks((1, 2, 3))
    ax2b.set_yticklabels((1, 10, 100))
    ax2b.set_yticks(np.log10((2, 5, 20, 50)) + 1, minor=True)
    ax2b.set_yticklabels((2, 5, 20, 50), minor=True)
    ax2b.grid(axis='y', which='major', linewidth=1)
    ax2b.grid(axis='y', which='minor')

    ax1.set_xticks(np.arange(0, 86401, 7200))
    ax2b.set_xticklabels(['00:00', '02:00', '04:00', '06:00',
                          '08:00', '10:00', '12:00', '14:00',
                          '16:00', '18:00', '20:00', '22:00', '24:00'])

    ax2b.grid(axis='x')

    if (pick_peak):
        periods = 1./(f+1e-10)
        width = 4
        peaks_periods, peakvals = pick_peaks(periods, slog, 
                                             plot=False,
                                             f_min=3.5, f_max=8.)
        ax2b.plot(t, np.log10(peaks_periods) + 1, 'g', linewidth=1.5)
        ax2b.plot(t, np.log10(peaks_periods - width/2) + 1, 'g--', linewidth=1.)
        ax2b.plot(t, np.log10(peaks_periods + width/2) + 1, 'g--', linewidth=1.)

        disp_int = _integrate_displacement(t, f, s, peaks_periods, width)

        # Write Picks to file
        write_picks(os.path.join(path_out, 'Picks'),
                    t, peaks_periods, peakvals, tr.stats)
        
        if tr.stats.channel[1] == 'D':
            ax1b.plot(t, disp_int/10, label='integrated displacement', color='r')
            ax1b.set_ylabel('Pressure / Pa', color='r')
        else:
            ax1b.plot(t, disp_int*1e6, label='integrated displacement', color='r')
            ax1b.set_ylabel('Displacement / m', color='r')

        ax1b.plot(t, peaks_periods, label='weighted peak', color='g')
        ax1b.tick_params('y', colors='r')
        ax1b.set_ylim(0, 10)

    if (pick_harmonics):
        for time, freq in zip(times, freqs):
            ax2b.plot(time, np.log10(1./freq) + 1, 'k+')
            ax2b.hlines((np.log10(1. / fmin_pick) + 1,
                         np.log10(1. / fmax_pick) + 1),
                        0, t[-1],
                        colors='k', linestyles='dashed')

        times_lp, level_lp = _pick_longperiod(t, f, s)

        for time, level in zip(times_lp, level_lp):
            posx = time - winlen/4
            posy = np.log10(1./fmin) + 1
            patchi = patches.Rectangle((posx, posy),
                                       width=winlen / 2, height=2,
                                       alpha=0.1,
                                       color=(0.0, 0, 0.8), edgecolor=None)
            ax2b.add_patch(patchi)
    if cat:
        _mark_events(cat, tr.stats, ax2b, ypos=2.9)

    # Axis with colorbar
    mappable = ax2b.collections[0]
    cb = plt.colorbar(mappable=mappable, cax=ax3)
    ax3.set_ylabel('Amplitude / dB')
    cb.set_ticks((-250, -200, -150, -100, -50, 0, 50, 100))

    # Axis with mean PSD of signal
    if plot_highfreq:
        ax4a.plot(np.mean(slog, axis=1), f, color='black')
        ax4a.plot(np.percentile(slog, axis=1, q=95), f,
                  color='darkgrey', linestyle='dashed')
        ax4a.plot(np.percentile(slog, axis=1, q=5), f,
                  color='darkgrey', linestyle='dashed')
        ax4a.set_ylim(1, f[-1])
        ax4a.set_ylabel('frequency / Hz')
        ax4a.yaxis.tick_right()
        ax4a.yaxis.set_label_position('right')
        ax4a.set_xticks((-250, -200, -150, -100, -50, 0, 50, 100))
        ax4a.set_xticklabels(())
        ax4a.set_xlim(vmin, vmax)
        ax4a.grid(axis='y', which='major', linewidth=1)
        ax4a.grid('on')

    ax4b.plot(np.mean(slog, axis=1), np.log10(1./f) + 1., color='black')
    ax4b.plot(np.percentile(slog, axis=1, q=95), np.log10(1./f) + 1.,
              color='darkgrey', linestyle='dashed')
    ax4b.plot(np.percentile(slog, axis=1, q=5), np.log10(1./f) + 1.,
              color='darkgrey', linestyle='dashed')
    ax4b.set_xticks((-250, -200, -150, -100, -50, 0, 50, 100))
    ax4b.set_xlim(vmin, vmax)
    ax4b.set_ylabel('period / seconds')
    ax4b.set_xlabel('Amplitude / dB')
    ax4b.yaxis.set_label_position('right')

    ax4b.set_yticks((1, 2, 3))
    ax4b.set_yticklabels((1, 10, 100))
    ax4b.set_yticks(np.log10((2, 5, 20, 50)) + 1, minor=True)
    ax4b.set_yticklabels((2, 5, 20, 50), minor=True)
    ax4b.grid(axis='y', which='major', linewidth=1)
    ax4b.grid(axis='y', which='minor')
    ax4b.grid(axis='x')
    ax4b.yaxis.tick_right()

    fig.suptitle('%s.%s Day %s - %s' %
                 (tr.stats.station,
                  tr.stats.channel,
                  tr.stats.starttime.strftime('%Y/%m/%d-%H:%M:%S'),
                  tr.stats.endtime.strftime('%Y/%m/%d-%H:%M:%S')))

    plt.savefig(os.path.join(path_out, 'Spectrograms', fnam_pic),
                dpi=dpi)
    plt.close('all')

    return f, t, s


def _mark_events(cat, stats, ax, ypos, stlat=54.7, stlon=12.7):
    # Plots an event marker
    from obspy.geodetics import locations2degrees
    for ev in cat:
        origin = ev.origins[0]
        desc = ev.event_descriptions[0]
        mag = ev.magnitudes[0]

        if (stats.starttime < origin.time < stats.endtime):
            dist = locations2degrees(stlat, stlon,
                                     origin.latitude, origin.longitude)
            region = desc['text']
            text = 'M%3.1f, %d deg, \n%s' % (mag.mag, dist, region)
            xpos = float(time_of_day(origin.time))
            ax.text(xpos-100, ypos, text,
                    color='darkblue', rotation='vertical', fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right')
            ax.vlines(ymin=0, ymax=4, x=xpos, color='darkblue',
                      linestyle='dashed', linewidth=2)

