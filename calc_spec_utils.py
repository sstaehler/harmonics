import obspy
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
from obspy.signal.util import next_pow_2
import harmonics
import matplotlib.patches as patches
import matplotlib as mpl

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


def write_picks(path, times, picks, stats, st_current):
    fnam_out = os.path.join(path, 'picks_%s_%s.txt' %
                            (stats.station, stats.channel))
    dt_curr = st_current[0].stats.delta
    st_curr_16m = st_current.select(station='16m')
    st_curr_18m = st_current.select(station='18m')
    with open(fnam_out, 'a') as fid:
        for t, p in zip(times, picks):
            t_out = stats.starttime + t
            curr_16m_trim = st_curr_16m.slice(starttime=t_out - dt_curr,
                                              endtime=t_out + dt_curr)
            data_16m = np.sqrt(curr_16m_trim[0].data[0]**2 +
                               curr_16m_trim[1].data[0]**2) * 1e-3
            curr_18m_trim = st_curr_18m.slice(starttime=t_out - dt_curr,
                                              endtime=t_out + dt_curr)
            data_18m = np.sqrt(curr_18m_trim[0].data[0]**2 +
                               curr_18m_trim[1].data[0]**2) * 1e-3
            fid.write('%s, %f, %f, %f\n' % (t_out, p, data_16m, data_18m))


def time_of_day(time):
    return time.hour*3600 + time.minute*60 + time.second


def calc_spec(fnam_smgr, fmin=1e-2, fmax=10, vmin=-180, vmax=-80, winlen=300,
              pick_harmonics=False, fmin_pick=0.4, fmax_pick=1.2,
              s_threshold=-140, path_out='.', nharms=6, sigma_min=1e-3,
              p_peak_min=5, st_current=None, cat=None):

    st = obspy.read(fnam_smgr)
    if len(st) > 1:
        raise ValueError('Only one trace per seismogram file, please!')
    else:
        tr = st[0]

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
    if st_current:
        ax1b = ax1.twinx()
    # Spectrogram axis (f>1Hz)
    ax2a = fig.add_axes([0.1, 0.35, 0.7, 0.4], sharex=ax1)
    # Spectrogram axis (p>1s)
    ax2b = fig.add_axes([0.1, 0.1, 0.7, 0.25], sharex=ax1)
    # Colorbar axis
    ax3 = fig.add_axes([0.83, 0.8, 0.03, 0.15])
    # Mean PSD axis (f>1Hz)
    ax4a = fig.add_axes([0.8, 0.35, 0.1, 0.4], sharey=ax2a)
    # Mean PSD axis (p>1s)
    ax4b = fig.add_axes([0.8, 0.1, 0.1, 0.25], sharey=ax2b)

    # Axis with seismogram
    ax1.plot(tr.times() + time_of_day(tr.stats.starttime), tr.data, 'k')
    #ymaxmin = np.percentile(abs(tr.data), q=99.5) * 1.1
    ymaxmin = max(abs(tr.data)) * 1.1
    ax1.set_ylim(-ymaxmin, ymaxmin)
    if tr.stats.channel == 'BDH':
        ax1.set_ylabel('Pressure / Pa')
    else:
        ax1.set_ylabel('Displacement / m')
    ax1.grid('on')
    plt.setp(ax1.get_xticklabels(), visible=False)

    # # Plot mean energy above fmax_pick
    # b = np.array([f[:] > fmax_pick,
    #               f[:] < fmax_pick*3]).all(axis=0)
    # ax1b.hlines(s_threshold, xmin=0, xmax=86400, linestyle='dashed', color='r')
    # ax1b.plot(t, np.mean(slog[b, :], axis=0), 'r')
    # ax1b.set_ylim(vmin*1.1, vmax*0.9)

    # If we ahave a current object, plot it in here.
    if st_current:
        data_x = st_current[0].times() - float(tr.stats.starttime) \
            + float(st_current[0].stats.starttime)

        # Plot current in 18 m depth
        st_curr_18m = st_current.select(station='18m')
        data_y = np.sqrt(st_curr_18m[0].data**2 +
                         st_curr_18m[1].data**2) * 1e-3
        ax1b.plot(data_x, data_y,
                  color='r', linewidth=2, label='18 meter')

        # Plot current in 18 m depth
        st_curr_16m = st_current.select(station='16m')
        data_y = np.sqrt(st_curr_16m[0].data**2 +
                         st_curr_16m[1].data**2) * 1e-3
        ax1b.plot(data_x, data_y, label='16 meter',
                  color='orange', linewidth=2)

        ax1b.set_ylim(0, 1)
        ax1b.set_ylabel('Current velocity, m/s', color='r')
        ax1b.tick_params('y', colors='r')
        ax1b.legend()


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
                    times, freqs, tr.stats, st_current)

    # Axis with low frequency part of spectrogram
    ax2b.pcolormesh(t, np.log10(1./f) + 1., slog,
                    vmin=vmin, vmax=vmax, cmap='plasma')
    ax2b.set_ylim(np.log10(1./fmin), 1)
    ax2b.set_xlabel('time / seconds')
    ax2b.set_ylabel('period / seconds')
    ax2b.set_xlim(0, t[-1])
    ax2b.set_yticks((1, 2, 3))
    ax2b.set_yticklabels((1, 10, 100))

    ax1.set_xticks(np.arange(0, 86401, 10800))
    ax2b.set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00',
                          '15:00', '18:00', '21:00', '24:00'])
    ax2b.grid('on')

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
    mappable = ax2a.collections[0]
    cb = plt.colorbar(mappable=mappable, cax=ax3)
    ax3.set_ylabel('Amplitude / dB')
    cb.set_ticks((-150, -100, -50, 0, 50, 100))

    # Axis with mean PSD of signal
    ax4a.plot(np.mean(slog, axis=1), f, color='black')
    ax4a.plot(np.percentile(slog, axis=1, q=95), f,
              color='darkgrey', linestyle='dashed')
    ax4a.plot(np.percentile(slog, axis=1, q=5), f,
              color='darkgrey', linestyle='dashed')
    ax4a.set_ylim(1, f[-1])
    ax4a.set_ylabel('frequency / Hz')
    ax4a.yaxis.tick_right()
    ax4a.yaxis.set_label_position('right')
    ax4a.set_xticks((-150, -100, -50, 0, 50, 100))
    ax4a.set_xticklabels(())
    ax4a.set_xlim(vmin, vmax)
    ax4a.grid('on')

    ax4b.plot(np.mean(slog, axis=1), np.log10(1./f) + 1., color='black')
    ax4b.plot(np.percentile(slog, axis=1, q=95), np.log10(1./f) + 1.,
              color='darkgrey', linestyle='dashed')
    ax4b.plot(np.percentile(slog, axis=1, q=5), np.log10(1./f) + 1.,
              color='darkgrey', linestyle='dashed')
    ax4b.set_yticks((1, 2, 3))
    ax4b.set_yticklabels((1, 10, 100))
    ax4b.set_xticks((-150, -100, -50, 0, 50, 100))
    ax4b.set_xlim(vmin, vmax)
    ax4b.set_xlabel('Amplitude / dB')
    ax4b.grid('on')
    ax4b.yaxis.tick_right()

    fig.suptitle('%s.%s Day %04d/%02d/%02d - %02d:%02d:%02d' %
                 (tr.stats.station,
                  tr.stats.channel,
                  tr.stats.starttime.year,
                  tr.stats.starttime.month,
                  tr.stats.starttime.day,
                  tr.stats.starttime.hour,
                  tr.stats.starttime.minute,
                  tr.stats.starttime.second,))

    plt.savefig(os.path.join(path_out, 'Spectrograms', fnam_pic),
                dpi=100)
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

