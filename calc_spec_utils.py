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


def write_picks(path, times, picks, stats):
    fnam_out = os.path.join(path, 'picks_%s_%s.txt' %
                            (stats.station, stats.channel))
    with open(fnam_out, 'a') as fid:
        for t, p in zip(times, picks):
            fid.write('%s, %f\n' % (stats.starttime + t, p))


def calc_spec(fnam_smgr, fmin=1e-2, fmax=10, vmin=-180, vmax=-80, winlen=300,
              pick_harmonics=True, fmin_pick=0.4, fmax_pick=1.2,
              s_threshold=-140, path_out='.', nharms=6, sigma_min=1e-3,
              p_peak_min=5., plot=False, file_out=False):

    st = obspy.read(fnam_smgr)
    if len(st) > 1:
        raise ValueError('Only one trace per seismogram file, please!')
    else:
        tr = st[0]

    tr.decimate(int(tr.stats.sampling_rate / 25))

    # Check whether spectrogram image already exists
    fnam_pic = '%s.%s_%02d_%02d_%02d.png' % (tr.stats.station,
                                             tr.stats.channel,
                                             tr.stats.starttime.month,
                                             tr.stats.starttime.day,
                                             tr.stats.starttime.hour)

    # if os.path.exists(os.path.join(path_out, 'Spectrograms', fnam_pic)):
    #     print('Image already exists')
    #     return

    s, f, t = specgram(tr.data, Fs=tr.stats.sampling_rate,
                       NFFT=winlen*tr.stats.sampling_rate,
                       pad_to=next_pow_2(winlen*tr.stats.sampling_rate) * 2,
                       noverlap=winlen*tr.stats.sampling_rate*0.5)
    slog = 10 * np.log10(s)
    f += 1e-10

    fig = plt.figure(figsize=(16, 8))

    # Seismogram axis
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax1b = ax1.twinx()
    # Spectrogram axis (f>1Hz)
    ax2a = fig.add_axes([0.1, 0.35, 0.7, 0.4], sharex=ax1)
    # Spectrogram axis (p>1s)
    ax2b = fig.add_axes([0.1, 0.1, 0.7, 0.25], sharex=ax1)
    # Colorbar axis
    ax3 = fig.add_axes([0.87, 0.8, 0.03, 0.15])
    # Mean PSD axis (f>1Hz)
    ax4a = fig.add_axes([0.8, 0.35, 0.1, 0.4], sharey=ax2a)
    # Mean PSD axis (p>1s)
    ax4b = fig.add_axes([0.8, 0.1, 0.1, 0.25], sharey=ax2b)

    # Axis with seismogram
    ax1.plot(tr.times(), tr.data, 'k')
    ymaxmin = max(abs(tr.data)) * 1.1  # np.percentile(abs(tr.data), q=99)
    ax1.set_ylim(-ymaxmin, ymaxmin)
    if tr.stats.channel == 'BDH':
        ax1.set_ylabel('Pressure / Pa')
    else:
        ax1.set_ylabel('Displacement / m')
    ax1.grid('on')

    # Plot mean energy above fmax_pick
    b = np.array([f[:] > fmax_pick,
                  f[:] < fmax_pick*3]).all(axis=0)
    ax1b.hlines(s_threshold, xmin=0, xmax=86400, linestyle='dashed', color='r')
    ax1b.plot(t, np.mean(slog[b, :], axis=0), 'r')
    ax1b.set_ylim(vmin*1.1, vmax*0.9)

    # Axis with high frequency part of spectrogram
    ax2a.pcolormesh(t, f, slog,
                    vmin=vmin, vmax=vmax, cmap='viridis')
    ax2a.set_ylim(1, f[-1])
    ax2a.set_ylabel('frequency / Hz')
    ax2a.grid('on')
    ax2a.hlines((fmin_pick, fmax_pick), 0, t[-1],
                colors='k', linestyles='dashed')

    if (pick_harmonics):
        times, freqs, peakvals, fmax_used = \
            harmonics.pick_spectrogram(f, t, s,
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

        write_picks(os.path.join(path_out, 'Picks'),
                    times, freqs, tr.stats)
        ax2a.plot(t, fmax_used, 'r')

    # Axis with low frequency part of spectrogram
    ax2b.pcolormesh(t, np.log10(1./f) + 1., slog,
                    vmin=vmin, vmax=vmax, cmap='viridis')
    ax2b.set_ylim(np.log10(1./fmin), 1)
    ax2b.set_xlabel('time / seconds')
    ax2b.set_ylabel('period / seconds')
    ax2b.set_xlim(0, t[-1])
    ax2b.set_yticks((1, 2, 3))
    ax2b.set_yticklabels((1, 10, 100))
    ax2b.grid('on')
    ax2b.plot(t, fmax_used, 'r')

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

    # Axis with colorbar
    mappable = ax2a.collections[0]
    plt.colorbar(mappable=mappable, cax=ax3)
    ax3.set_ylabel('Amplitude $10*log_{10}((m/s^2)^2/Hz)$')

    # Axis with mean PSD of signal
    ax4a.plot(np.mean(slog, axis=1), f, color='black')
    ax4a.plot(np.percentile(slog, axis=1, q=95), f,
              color='darkgrey', linestyle='dashed')
    ax4a.plot(np.percentile(slog, axis=1, q=5), f,
              color='darkgrey', linestyle='dashed')
    ax4a.set_ylim(1, f[-1])
    ax4a.set_ylabel('frequency / Hz')
    ax4a.yaxis.tick_right()
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

    ax1.set_title('%s.%s Day %02d/%02d - %02d:%02d:%02d' %
                  (tr.stats.station,
                   tr.stats.channel,
                   tr.stats.starttime.month,
                   tr.stats.starttime.day,
                   tr.stats.starttime.hour,
                   tr.stats.starttime.minute,
                   tr.stats.starttime.second,))
    # if plot:
    #     plt.show()

    # if file_out:
    #     fnam = '%s.%s_%02d_%02d_%02d.npy' % (tr.stats.station,
    #                                          tr.stats.channel,
    #                                          tr.stats.starttime.month,
    #                                          tr.stats.starttime.day,
    #                                          tr.stats.starttime.hour)
    #     np.savez_compressed(os.path.join(dirpath, 'Spectrograms', 'data', fnam),
    #                         s=s, f=f, t=t)

    #     plt.savefig(os.path.join(dirpath, 'Spectrograms', fnam_pic), dpi=300)

    # if pick_harmonics and file_out:
    #     fnam = os.path.join(dirpath, 'picks', tr.stats.station,
    #                         '%s_harmonic.txt' % tr.stats.channel)
    #     with open(fnam, 'a') as fid:
    #         for time, freq, peakval in zip(times, freqs, peakvals):
    #             t_string = str(tr.stats.starttime + time)
    #             fid.write('%s, %f, %f\n' % (t_string, freq, peakval))

    #     fnam = os.path.join(dirpath, 'picks', tr.stats.station,
    #                         '%s_long_period.txt' % tr.stats.channel)
    #     with open(fnam, 'a') as fid:
    #         for time, level in zip(times_lp, level_lp):
    #             t_string = str(tr.stats.starttime + time)
    #             fid.write('%s, %f\n' % (t_string, level))

    plt.savefig(os.path.join(path_out, 'Spectrograms', fnam_pic), dpi=200)
    plt.close('all')

    return f, t, s
