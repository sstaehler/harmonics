#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------
#   Filename:  find_harmonics.py
#   Purpose:   Find harmonic signal in a seismogram
#   Author:    Simon Staehler
#   Email:     mail@simonstaehler.com
#   License:   GNU Lesser General Public License, Version 3
# -------------------------------------------------------------------

from tqdm import tqdm
import argparse
import glob
from calc_spec_utils import calc_spec


def define_arguments():
    helptext = 'Find harmonic signal in seismogram'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = 'Path to Seismogram files (can be anything that Obspy reads)'
    parser.add_argument('--smgr_path', help=helptext)

    helptext = 'Output directory'
    parser.add_argument('--out_path', help=helptext, default='.')

    return parser


def main():
    parser = define_arguments()

    # Parse input arguments
    args = parser.parse_args()

    files = glob.glob(args.smgr_path)
    if (len(files) == 0):
        print('Did not find any files in path: %s' % args.smgr_path)
        raise ValueError()

    files.sort()

    for fnam_smgr in tqdm(files):
        calc_spec(fnam_smgr, fmin=1e-2, fmax=10,
                  vmin=-150, vmax=-80, winlen=300,
                  pick_harmonics=True, fmin_pick=0.4, fmax_pick=4.0,
                  s_threshold=-160,
                  path_out=args.out_path)

if __name__ == "__main__":
    main()
