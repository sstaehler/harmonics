import glob
import os
from calc_spec_utils import calc_spec
from multiprocessing import Pool
from tqdm import tqdm
from obspy import read_events
import sys

dirpath = '/opt/DARS/Corrected/'
stat = sys.argv[1]

cat = read_events(sys.argv[2])

for chan in ['?DH', '?H?']:
    files = glob.glob(os.path.join(dirpath, stat, chan, '*.seed'))
    files.sort()

    if __name__ == '__main__':
        for file in tqdm(files):
            if chan=='?DH':
                vmin=-60
                vmax=40
            else:
                vmin=-160
                vmax=-60

            calc_spec(file, cat=cat, vmin=vmin, vmax=vmax)


#    p = Pool(4)
#    for _ in tqdm(p.imap_unordered(calc_spec, files),
#                  total=len(files)):
#        pass
    # p.map(calc_spec, files)
