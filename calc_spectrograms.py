import glob
import os
from calc_spec_utils import calc_spec
from multiprocessing import Pool

dirpath = '/media/staehler/Staehler_Transfer/Noise LOBSTER/'
stat = 'TDC01'
# chan = 'HHX'

fmin = 1e-2
fmax = 10.

vmin = -180.
vmax = -80.

winlen = 300.

files = glob.glob(os.path.join(dirpath, stat, 'HH[XYZ]', '*.mseed'))
files.sort()

if __name__ == '__main__':
    p = Pool(4)
    p.map(calc_spec, files)
