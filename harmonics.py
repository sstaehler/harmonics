import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from obspy.signal.util import next_pow_2
from scipy.signal import decimate, detrend
from scipy.optimize import minimize
from scipy import convolve


def _gaussfun(x, x0, p_peak, sigma, baseline=0.0, basegrad=0.0):
    return p_peak * np.exp(- (x - x0) ** 2 / (2 * sigma) ** 2) + \
           baseline + basegrad * x


def _misfit(m, f, s):
    x0 = m[0]
    p_peak = m[1]
    sigma = m[2]
    baseline = 0.0  # m[3]
    basegrad = 0.0  # m[4]
    diff = _gaussfun(f, x0, p_peak, sigma, baseline, basegrad) - s
    return np.sqrt(np.sum(diff**2))


def pick_spectrogram(f, t, s, fwin=(0.4, 1.1), winlen=150, sigma_min=0.005):
    times = []
    freqs = []
    times_cleaned = []
    freqs_cleaned = []

    for i in range(0, len(t)):
        f_peak, p_peak, sigma = HPS(f, s[:, i],
                                    fwin_pick=fwin,
                                    maxharms=3, plot=False)

        if (sigma > sigma_min and fwin[0] < f_peak < fwin[1]):
            freqs.append(f_peak)
            times.append(t[i])

    for i in range(0, len(freqs)-1):
        if times[i+1] - times[i] == winlen:
            times_cleaned.append(times[i])
            freqs_cleaned.append(freqs[i])

    return times_cleaned, freqs_cleaned


def freq_from_HPS(tr, winlen=120., fwin_pick=(1., 4.), plot=False, maxharms=8):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    """

    # harmonic product spectrum:
    c, f = mlab.psd(tr.data,
                    Fs=tr.stats.sampling_rate,
                    NFFT=winlen*tr.stats.sampling_rate,
                    pad_to=next_pow_2(winlen*tr.stats.sampling_rate))
    c = 10*np.log10(c)

    f_peak, p_peak = HPS(f, c, fwin_pick, plot, maxharms)
    return f_peak, p_peak


def HPS(f, c, fwin_pick=(1., 4.), plot=False, maxharms=8):
    c_prod = np.copy(c[0: int(len(c) / maxharms)])
    c_prod = convolve(c_prod, [0.25, 0.5, 0.25], mode='same')

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(f,
                c,
                label='Main mode')

    for i in range(2, maxharms + 1):
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

    # Pick maximum by fitting Gaussian
    # Set starting values
    # f0 = np.mean(fwin_pick)
    x0 = np.asarray((f_peak_initial, p_peak_initial, 0.01)) # -126, -7))

    # minimize
    res = minimize(fun=_misfit, x0=x0,
                   args=(f_prod, c_prod_det),
                   bounds=((0, None),
                           (None, None),
                           (0, None))
                   )
    # print(_misfit(m=res['x'], f=f_prod, s=c_prod_det))
    f_peak = res['x'][0]
    p_peak = res['x'][1]
    sigma = res['x'][2]
    # print(res['success'])
    # print(res['message'])
    # print('f_peak:', f_peak)
    # print('p_peak:', p_peak)
    # print('sigma :', sigma)
    if plot:
        ax.plot(f_prod,
                c_prod,
                'k',
                label='Product', LineWidth=3)
        ax.plot(f_prod, c_prod_det - 140, label='detrend',
                LineWidth=3)
        ax.plot(f_prod,
                _gaussfun(f_prod, *x0) - 140,
                'g', Linewidth=3,
                label='X0')
        ax.plot(f_prod,
                _gaussfun(f_prod, *res['x']) - 140,
                'r', Linewidth=3,
                label='fit')
        ax.set_xlim(f[0], f[-1]/maxharms)
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
