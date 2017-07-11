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

    helptext = 'Number of harmonics'
    parser.add_argument('--nharms', help=helptext, type=int, default=4)

    helptext = 'Frequency range (minimum)'
    parser.add_argument('--fmin', help=helptext, type=float, default=0.4)

    helptext = 'Frequency range (maximum)'
    parser.add_argument('--fmax', help=helptext, type=float, default=4.0)

    helptext = 'Plot range in dB (maximum), choose larger value for' + \
            'hydrophone channel (i.e. 40)'
    parser.add_argument('--vmax', help=helptext, type=float, default=-60.0)

    helptext = 'Plot range in dB (minimum), choose larger value for' + \
            'hydrophone channel (i.e. -60)'
    parser.add_argument('--vmin', help=helptext, type=float, default=-160.0)

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
                  vmin=args.vmin, vmax=args.vmax, winlen=300,
                  pick_harmonics=True, 
                  fmin_pick=args.fmin, fmax_pick=args.fmax,
                  s_threshold=-160, nharms=args.nharms, dpi=300,
                  path_out=args.out_path)

if __name__ == "__main__":
    main()
