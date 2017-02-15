import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from obspy.signal.util import next_pow_2
from scipy.signal import decimate, detrend
from scipy.optimize import minimize
from scipy import convolve


def _gaussfun(x, x0, p_peak, sigma):
    return p_peak * np.exp(- (x - x0) ** 2 / (2 * sigma) ** 2)


def _misfit(m, f, s):
    x0 = m[0]
    p_peak = m[1]
    sigma = m[2]
    diff = _gaussfun(f, x0, p_peak, sigma) - s
    return np.sqrt(np.sum(diff**2))


def pick_spectrogram(f, t, s, fwin=(0.4, 1.1), winlen=150, sigma_min=0.005,
                     p_peak_min=10.0, s_threshold=-150,
                     nharms=4, plot=False, verbose=False):
    times = []
    freqs = []
    p_peaks = []

    fmax_used = []

    f_prop = -1

    # Calculate mean energy between fmax and 3*fmax
    b = np.array([f[:] > fwin[1],
                  f[:] < fwin[1]*3]).all(axis=0)
    s_mean = np.mean(s[b, :], axis=0)

    for i in range(0, len(t)):
        if s_mean[i] > s_threshold:
            f_peak, p_peak, sigma = HPS(f, s[:, i],
                                        fwin_pick=fwin,
                                        nharms=nharms,
                                        f_prop=f_prop,
                                        plot=plot)

            if f_peak > (fwin[1] - fwin[0]) / 2:
                f_peak_2, p_peak_2, sigma_2 \
                    = HPS(f, s[:, i],
                          fwin_pick=fwin,
                          nharms=1,
                          f_prop=f_peak / 2,
                          plot=plot)

                choose_new = sigma_2 > sigma_min and \
                    p_peak_2 > p_peak * 0.8 and \
                    f_peak_2 < f_peak * 0.75

                if choose_new:
                    f_peak = f_peak_2
                    p_peak = p_peak_2
                    sigma = sigma_2

                if verbose:
                    print(t[i], f_peak, f_peak_2,
                          p_peak, p_peak_2, sigma_2, choose_new)

            pick = (sigma > sigma_min and
                    fwin[0] < f_peak < fwin[1] and
                    p_peak > p_peak_min)

            if verbose:
                print(t[i], f_peak, p_peak, sigma, pick)

            if pick:
                freqs.append(f_peak)
                times.append(t[i])
                p_peaks.append(p_peak)

            if pick:
                f_prop = f_peak
                # fwin[1] = f_prop * 1.1
            else:
                f_prop = -1
                # fwin = fwin_orig

            fmax_used.append(fwin[1])

    return times, freqs, p_peaks, fmax_used


def freq_from_HPS(tr, winlen=120., fwin_pick=(1., 4.), nharms=6, f_prop=-1,
                  plot=False):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    """

    # harmonic product spectrum:
    c, f = mlab.psd(tr.data,
                    Fs=tr.stats.sampling_rate,
                    NFFT=winlen*tr.stats.sampling_rate,
                    pad_to=next_pow_2(winlen*tr.stats.sampling_rate))
    c = 10*np.log10(c)

    f_peak, p_peak = HPS(f, c, fwin_pick, plot, nharms)
    return f_peak, p_peak


def HPS(f, c, fwin_pick, plot=False, nharms=6, f_prop=-1):
    c_prod = np.copy(c[0: int(len(c) / nharms)])
    c_prod = convolve(c_prod, [0.25, 0.5, 0.25], mode='same') * 2

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(f,
                c,
                label='Main mode')

    for i in range(2, nharms + 1):
        a = decimate(c, i)
        c_prod += a[0:len(c_prod)]
        if plot:
            ax.plot(f[0:len(a)],
                    a,
                    label='Harm %d (a)' % i)

    f_red = f[0:len(c_prod)]
    b = np.array([f_red > fwin_pick[0],
                  f_red < fwin_pick[1]]).all(axis=0)

    # Reduce to values in frequency range
    f_prod = f_red[b]
    c_prod = c_prod[b]

    # detrend c_prod
    c_prod_det = detrend(c_prod)

    # pick maximum value
    i = np.argmax(c_prod_det)
    f_peak_initial = f_prod[i]
    p_peak_initial = c_prod_det[i]

    # if a frequency value has been proposed, override the maximum
    if fwin_pick[0] < f_prop < fwin_pick[1]:
        f_peak_initial = f_prop

    # Pick maximum by fitting Gaussian
    # Set starting values
    x0 = np.asarray((f_peak_initial, p_peak_initial, 0.004))

    # minimize
    res = minimize(fun=_misfit, x0=x0,
                   args=(f_prod, c_prod_det),
                   bounds=((0, None),
                           (None, None),
                           (0, None)))
    f_peak = res['x'][0]
    p_peak = res['x'][1]
    sigma = res['x'][2]

    if plot:
        # ax.plot(f_prod,
        #         c_prod,
        #         'k',
        #         label='Product', LineWidth=3)
        ax.plot(f_prod, c_prod_det - 140, 'k',
                label='detrend',
                LineWidth=3)
        ax.plot(f_prod,
                _gaussfun(f_prod, *x0) - 140,
                'g', Linewidth=3,
                label='X0')
        ax.plot(f_prod,
                _gaussfun(f_prod, *res['x']) - 140,
                'r', Linewidth=3,
                label='fit')
        ax.set_xlim(f[0], f[-1]/nharms)
        ax.legend()
        ax.vlines(fwin_pick[0], -150, -90, 'r')
        ax.vlines(fwin_pick[1], -150, -90, 'r')
        ax.vlines(f_peak, -150, -90, 'k')
        ax.vlines(f_peak_initial, -150, -90)
        ax.set_ylim((-150, -90))
        plt.show()
    return f_peak, p_peak, sigma


def parabolic_fit(x, y, i_guess, winlen):
    win_start = int(max(i_guess - winlen/2, 0))
    win_end = int(min(i_guess + winlen/2, len(x))) + 1
    x_vals = x[win_start:win_end]
    y_vals = y[win_start:win_end]
    r = np.polyfit(x_vals, y_vals, 2)
    return -r[1] / 2 / r[0], (4*r[0]*r[2] - r[1]**2) / 4*r[0], r
