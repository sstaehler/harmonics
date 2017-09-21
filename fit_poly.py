import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def _parabolic_fit(x, y, i_guess, winlen, ax=None):
    win_start = int(max(i_guess - winlen/2, 0))
    win_end = int(min(i_guess + winlen/2, len(x))) + 1
    x_vals = x[win_start:win_end]
    y_vals = y[win_start:win_end]
    if (ax):
        ax.plot(x_vals, y_vals, 'k', linewidth=2.5)
    r = np.polyfit(x_vals, y_vals, 2)
    return -r[1] / 2 / r[0], r[2] - r[1]**2 / (4*r[0]), r


def pick_peak(f_in, p_in, ax=None):
    i = np.argmax(p_in)

    f_peak_initial = f_in[i]
    p_peak_initial = p_in[i]

    if (ax):
        a, b, r = _parabolic_fit(f_in, p_in, i, 500, ax)
        ax.plot(f_peak_initial, p_peak_initial, 'ro')
        ax.plot(a, b, 'go')
        ax.plot(f_in, r[0] * f_in**2 + r[1] * f_in + r[2])
    else:
        a, b, r = _parabolic_fit(f_in, p_in, i, 500)

    res = [a, b, r]
    return res


def weighted_mean(x, w, exp=1.0):
    w_norm = w - np.min(w)
    w_norm /= np.sum(w_norm**exp) 
    return np.sum(x * w_norm**exp)


def pick_peaks(f_in, p_in, f_min=1., f_max=10., plot=False):
    f_peaks = np.zeros(p_in.shape[1])
    p_peaks = np.zeros(p_in.shape[1])
    sigma_peaks = np.zeros(p_in.shape[1])

    i = 0

    bol = np.array((f_in>f_min, f_in<f_max)).all(axis=0)

    for i in range(0, p_in.shape[1]):
        if plot:
            fig, ax = plt.subplots(1,1)
            res = pick_peak(f_in[bol], p_in[bol, i], ax)
        else:
            res = pick_peak(f_in[bol], p_in[bol, i])

        mean = weighted_mean(f_in[bol], p_in[bol, i])

        f_peaks[i] = res[0]
        p_peaks[i] = res[1]

        if plot:
            ax.plot(f_in, p_in[:, i], label='spectrum')
            ax.plot(f_in[bol], p_in[bol, i], linewidth=1.5, color='r',
                    label='spectrum, active')
            ax.plot(mean, np.max(p_in), 'bo')
            #ax.set_ylim(-160, -60)
            ax.set_xlim(0, 20)
            fig.savefig('Fits/%05d.png' % i)
            plt.close()
            print(i, f_peaks[i], mean)


    return f_peaks, p_peaks


