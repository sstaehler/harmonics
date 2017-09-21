import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def _gaussfun(x, x0, p_peak, sigma, p_mean):
    return p_peak * np.exp(- (x - x0) ** 2 / (2 * sigma) ** 2) + p_mean


def _misfit(m, f, s):
    x0 = m[0]
    p_peak = m[1]
    sigma = m[2]
    p_mean = m[3]

    diff = _gaussfun(f, x0, p_peak, sigma, p_mean) - s
    return np.sum(diff**2) 


def _jakobian(m, f, s):
    f0 = m[0]
    p_peak = m[1]
    sigma = m[2]
    p_mean = m[3]

    diff = _gaussfun(f, f0, p_peak, sigma, p_mean) - s
    
    j = np.zeros(shape=(4, f.shape[0]))
    
    # Term in exponent, which is not changed in the derivatives
    expterm = np.exp(- (f - f0) ** 2 / (2 * sigma) ** 2)
    
    # derivative x0
    j[0, :] = p_peak * expterm * ((f - f0) / 2 / sigma**2 ) * (2 * diff)

    # derivative p_peak
    j[1, :] = expterm * (2 * diff)
    
    # derivative sigma
    j[2, :] = p_peak * expterm * (f - f0)**2 / (2 * sigma**3) * (2 * diff)

    # derivative p_peak
    j[3, :] = 1 * (2 * diff)
    
    return np.sum(j, axis=1) 

def pick_peak(f_in, p_in, ax=None):
    i = np.argmax(p_in)
    f_peak_initial = f_in[i]
    p_peak_initial = p_in[i]

    p_median = np.median(p_in)
    p_peak_initial -= p_median

    p_eff = p_in - np.mean(p_in)
    p_eff /= np.max(p_eff)

    f_mean = np.sum(p_eff*f_in) / np.sum(abs(p_eff))

    std = np.sqrt(np.sum((abs(p_eff)*(f_in-f_peak_initial))**2) / np.sum(abs(p_eff)))
    x0 = np.asarray((f_peak_initial, 
                     p_peak_initial, 
                     std, 
                     p_median))
    print(x0, f_mean)

    if (ax):
        ax.plot(f_peak_initial, p_peak_initial+p_median, 'ro')
        ax.plot(f_mean, p_peak_initial+p_median, 'go')

    res = minimize(fun=_misfit, x0=x0,
		   args=(f_in, p_in),
		   bounds=((1e-20, None),
			   (1e-20, None),
			   (1e-20, None),
			   (1e-20, None)),
		   jac=_jakobian,
		   tol=1e-16)

    print(res['x'])

    # f_peak = res['x'][0]
    # p_peak = res['x'][1]
    # sigma = res['x'][2]
    # p_mean = res['x'][3]

    return res['x']


def pick_peaks(t_in, f_in, p_in, plot=False):

    f_peaks = np.zeros(t_in.shape)
    p_peaks = np.zeros(t_in.shape)
    sigma_peaks = np.zeros(t_in.shape)

    i = 0

    for i in range(0, len(t_in)):
        if plot:
            fig, ax = plt.subplots(1,1)
            res = pick_peak(f_in, p_in[:, i], ax)
        else:
            res = pick_peak(f_in, p_in[:, i])

        f_peaks[i] = res[0]
        p_peaks[i] = res[1]
        sigma_peaks[i] = res[2]

        if plot:
            ax.plot(f_in, p_in[:, i], label='spectrum')
            p_pred = _gaussfun(f_in, res[0], res[1], res[2], res[3])
            ax.plot(f_in, p_pred, label='fit')
            ax.set_ylim(-160, -60)
            fig.savefig('Fits/%s.png' % t_in[i])
            plt.close()


    return f_peaks, p_peaks, sigma_peaks


