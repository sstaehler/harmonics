import obspy
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import specgram
from obspy.signal.util import next_pow_2
import harmonics
import matplotlib.patches as patches

import argparse
import glob
from progressbar import Percentage, ETA, ProgressBar, Bar, FileTransferSpeed


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


def calc_spec(file, fmin=1e-2, fmax=10, vmin=-180, vmax=-80, winlen=300,
              pick_harmonics=True, fmin_pick=0.4, fmax_pick=1.2):

    # Get Name of station and channel from file names (only works with the
    # TDC data).
    dirpath = '/'.join(file.split('/')[0:-3])

    inv = obspy.read_inventory(os.path.join(dirpath, 'TC.xml'))
    tr = obspy.read(file)[0]

    fnam_corr = os.path.join(dirpath, tr.stats.station, 'corr',
                             tr.stats.channel,
                             os.path.split(file)[1])

    if os.path.exists(fnam_corr):
        tr = obspy.read(fnam_corr)[0]

    else:
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
        times, freqs, peakvals = harmonics.pick_spectrogram(f, t, s,
                                                            fwin=(fmin_pick,
                                                                  fmax_pick))
        for time, freq in zip(times, freqs):
            posx = time - winlen/4
            posy = f[0]
            patchi = patches.Rectangle((posx, posy),
                                       width=winlen/2, height=f[-1],
                                       alpha=0.1,
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
    ax3.set_ylabel('Amplitude $10*log_{10}(m/s^2/Hz)$')

    ax1.set_title('%s.%s Day %02d/%02d - %02d:%02d:%02d' %
                  (tr.stats.station,
                   tr.stats.channel,
                   tr.stats.starttime.month,
                   tr.stats.starttime.day,
                   tr.stats.starttime.hour,
                   tr.stats.starttime.minute,
                   tr.stats.starttime.second,))

    plt.savefig(os.path.join(dirpath, 'Spectrograms', fnam_pic), dpi=300)
    plt.close('all')

    if pick_harmonics:
        fnam = os.path.join(dirpath, 'picks', tr.stats.station,
                            '%s_harmonic.txt' % tr.stats.channel)
        with open(fnam, 'a') as fid:
            for time, freq, peakval in zip(times, freqs, peakvals):
                t_string = str(tr.stats.starttime + time)
                fid.write('%s, %f, %f\n' % (t_string, freq, peakval))

        fnam = os.path.join(dirpath, 'picks', tr.stats.station,
                            '%s_long_period.txt' % tr.stats.channel)
        with open(fnam, 'a') as fid:
            for time, level in zip(times_lp, level_lp):
                t_string = str(tr.stats.starttime + time)
                fid.write('%s, %f\n' % (t_string, level))


# Main program
helptext = 'Plot spectrograms of OBS data and find harmonic signal'
parser = argparse.ArgumentParser(description=helptext)

helptext = 'Path to data files'
parser.add_argument('data_path', help=helptext)

helptext = 'Minimum frequency for harmonic picking (default: 0.4 Hz)'
parser.add_argument('--fmin', type=float, default=0.4)

helptext = 'Maximum frequency for harmonic picking (default: 1.2 Hz)'
parser.add_argument('--fmax', type=float, default=1.2)

args = parser.parse_args()

file_list = glob.glob(args.data_path)
file_list.sort()

# Create Progressbar widgets
widgets = ['Calculating: ', Percentage(), ' ', Bar(),
           ' ', ETA(), ' ', FileTransferSpeed()]

pbar = ProgressBar(widgets=widgets, max_value=len(file_list)).start()

for fnam in file_list:
    calc_spec(fnam, fmin_pick=args.fmin, fmax_pick=args.fmax)
    pbar += 1
