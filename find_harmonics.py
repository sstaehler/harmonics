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
import os
from calc_spec_utils import calc_spec


def define_arguments():
    helptext = 'Plot spectrogram and find spectral peaks in in seismogram'
    parser = argparse.ArgumentParser(description=helptext)

    helptext = 'Path to Seismogram files (can be anything that Obspy reads)'
    parser.add_argument('smgr_path', help=helptext)

    helptext = 'Kind of signal to find. Options: none, harmonic or peak'
    parser.add_argument('-k', '--kind', 
                        choices=['none', 'peak', 'harmonic'],
                        default='none',
                        help=helptext)

    helptext = 'Number of harmonics (default: 4)'
    parser.add_argument('--nharms', help=helptext, type=int, default=4)

    helptext = 'Minimum frequency in Hz (default: 0.4)'
    parser.add_argument('--fmin', help=helptext, type=float, default=0.4)

    helptext = 'Maximum frequency in Hz (default: 4.0)'
    parser.add_argument('--fmax', help=helptext, type=float, default=4.0)

    helptext = 'Minimum of plot range in dB (default: -160), choose larger value for' + \
            'hydrophone channel (i.e. -60)'
    parser.add_argument('--vmin', help=helptext, type=float, default=-160.0)

    helptext = 'Maximum of plot range in dB (default: -60), choose larger value for' + \
            'hydrophone channel (i.e. 40)'
    parser.add_argument('--vmax', help=helptext, type=float, default=-60.0)

    helptext = 'Window length for spectrogram calculation in seconds, default: 300'
    parser.add_argument('--winlen', help=helptext, type=float, default=300.0)

    helptext = 'Do not plot high frequency part of the seismogram (above 1 Hz)'
    parser.add_argument('--skip_hf', help=helptext, action='store_true', default=False)

    helptext = 'Output directory (default: ''.'')'
    parser.add_argument('--out_path', help=helptext, default='.')

    return parser


def main():
    # Define and parse input arguments
    parser = define_arguments()
    args = parser.parse_args()

    # Test whether output directories exist and create, if necessary
    dir_Picks = os.path.join(args.out_path, 'Picks')
    os.makedirs(dir_Picks, exist_ok=True)
    dir_Specs = os.path.join(args.out_path, 'Spectrograms')
    os.makedirs(dir_Specs, exist_ok=True)
    
    # Create list of all seismogram files
    files = glob.glob(args.smgr_path)
    if (len(files) == 0):
        print('Did not find any files in path: %s' % args.smgr_path)
        raise ValueError()
    files.sort()

    # Calc spectrograms and pick maxima for all seismogram files
    for fnam_smgr in tqdm(files):
        calc_spec(fnam_smgr, fmin=1e-2, fmax=10,
                  vmin=args.vmin, vmax=args.vmax, winlen=args.winlen,
                  plot_highfreq=not(args.skip_hf), 
                  pick_harmonics=(args.kind=='harmonic'), 
                  pick_peak=(args.kind=='peak'), 
                  fmin_pick=args.fmin, fmax_pick=args.fmax,
                  s_threshold=-160, nharms=args.nharms, dpi=300,
                  path_out=args.out_path)

if __name__ == "__main__":
    main()
