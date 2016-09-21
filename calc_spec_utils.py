import obspy
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
from obspy.signal.util import next_pow_2
import harmonics
import matplotlib.patches as patches


def calc_spec(file, fmin=1e-2, fmax=10, vmin=-180, vmax=-80, winlen=300,
              pick_harmonics=True, fmin_pick=0.4, fmax_pick=1.2):
    stat = file.split('/')[-3]
    chan = file.split('/')[-2]
    dirpath = '/'.join(file.split('/')[0:-3])

    fnam_corr = os.path.join(dirpath, stat, 'corr', chan,
                             os.path.split(file)[1])

    if os.path.exists(fnam_corr):
        tr = obspy.read(fnam_corr)[0]
        print(tr)

    else:
        inv = obspy.read_inventory(os.path.join(dirpath, 'TDC01.xml'))
        tr = obspy.read(file)[0]
        tr.stats.network = 'AW'
        if tr.stats.channel[2] == 'X':
            tr.stats.channel = 'HHN'
        elif tr.stats.channel[2] == 'Y':
            tr.stats.channel = 'HHE'

        print(tr)
        tr.attach_response(inv)
        tr.remove_response(pre_filt=(1./240., 1./180., 20., 25.))
        tr.decimate(2, no_filter=True)

    tr.write(fnam_corr, format='MSEED')

    # Check whether spectrogram image already exists
    fnam_pic = '%s.%s_%02d_%02d_%02d.png' % (tr.stats.station,
                                             tr.stats.channel,
                                             tr.stats.starttime.month,
                                             tr.stats.starttime.day,
                                             tr.stats.starttime.hour)

    if os.path.exists(os.path.join(dirpath, 'Spectrograms', fnam_pic)):
        print('Image already exists')
        return

    s, f, t = specgram(tr.data, Fs=tr.stats.sampling_rate,
                       NFFT=winlen*tr.stats.sampling_rate,
                       pad_to=next_pow_2(winlen*tr.stats.sampling_rate),
                       noverlap=winlen*tr.stats.sampling_rate*0.5)

    s = 10*np.log10(s)

    fnam = '%s.%s_%02d_%02d_%02d.npy' % (tr.stats.station,
                                         tr.stats.channel,
                                         tr.stats.starttime.month,
                                         tr.stats.starttime.day,
                                         tr.stats.starttime.hour)
    np.savez_compressed(os.path.join(dirpath, 'Spectrograms', fnam),
                        s=s, f=f, t=t)

    # setup figure
    fig = plt.figure()

    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax2a = fig.add_axes([0.1, 0.35, 0.7, 0.4], sharex=ax1)
    ax2b = fig.add_axes([0.1, 0.1, 0.7, 0.25], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

    # Axis with seismogram
    ax1.plot(tr.times(), tr.data, 'k')
    ax1.set_ylim(-1e-4, 1e-4)
    ax1.grid('on')

    # Axis with high frequency part of spectrogram
    ax2a.pcolormesh(t, f, s,
                    vmin=vmin, vmax=vmax, cmap='viridis')
    ax2a.set_ylim(1, f[-1])
    ax2a.set_ylabel('frequency / Hz')
    ax2a.grid('on')

    if (pick_harmonics):
        print('Picking harmonics:')
        times, freqs = harmonics.pick_spectrogram(f, t, s,
                                                  fwin=(fmin_pick, fmax_pick))
        for time, freq in zip(times, freqs):
            posx = time - winlen/4
            posy = f[0]
            patchi = patches.Rectangle((posx, posy),
                                       width=winlen/2, height=f[-1],
                                       alpha=0.2,
                                       color=(0.8, 0, 0), edgecolor=None)
            ax2a.add_patch(patchi)
            ax2a.plot(time, freq, 'k+')

        ax2a.hlines((fmin_pick, fmax_pick), 0, t[-1],
                    colors='k', linestyles='dashed')

    # Axis with low frequency part of spectrogram
    ax2b.pcolormesh(t, np.log10(1./f) + 1., s,
                    vmin=vmin, vmax=vmax, cmap='viridis')
    ax2b.set_ylim(np.log10(1./fmin), 1)
    ax2b.set_xlabel('time / seconds')
    ax2b.set_ylabel('period / seconds')
    ax2b.set_xlim(0, t[-1])
    ax2b.set_yticks((1, 2, 3))
    ax2b.set_yticklabels((1, 10, 100))
    ax2b.grid('on')

    if (pick_harmonics):
        for time, freq in zip(times, freqs):
            ax2b.plot(time, np.log10(1./freq) + 1, 'k+')
            ax2b.hlines((np.log10(1. / fmin_pick) + 1,
                         np.log10(1. / fmax_pick) + 1),
                        0, t[-1],
                        colors='k', linestyles='dashed')

        # pick time windows with high long-period noise level
        slope = []
        cross = []
        for i in range(0, len(t)):
            # Reduce to periods between 10 and 100s
            x = 1. / f[f < 0.1]
            y = s[f < 0.1, i]
            y = y[x < 100]
            x = x[x < 100]

            a, b = np.polyfit(x, y, deg=1)
            slope.append(a)
            cross.append(b)

        for time, crossi in zip(t, cross):
            if crossi > np.median(cross) + 5:
                posx = time - winlen/4
                posy = np.log10(1./freq) + 1
                patchi = patches.Rectangle((posx, posy),
                                           width=winlen / 2, height=2,
                                           alpha=0.2,
                                           color=(0.0, 0, 0.8), edgecolor=None)
                ax2b.add_patch(patchi)

    # Axis with colorbar
    mappable = ax2a.collections[0]
    plt.colorbar(mappable=mappable, cax=ax3)
    ax3.set_ylabel('Amplitude $10*log_{10}(m/s^2/Hz)$')

    ax1.set_title('%s.%s Day %02d/%02d - %02d:%02d:%02d' %
                  (tr.stats.station,
                   tr.stats.channel,
                   tr.stats.starttime.month,
                   tr.stats.starttime.day,
                   tr.stats.starttime.hour,
                   tr.stats.starttime.minute,
                   tr.stats.starttime.second,))

    plt.savefig(os.path.join(dirpath, 'Spectrograms', fnam_pic), dpi=200)
    plt.close('all')
