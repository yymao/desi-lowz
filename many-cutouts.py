#!/usr/bin/python3

"""
many-cutouts.py
---------------
Modified by: Yao-Yuan Mao (yymao) - Oct 22, 2025
Modified from: https://github.com/legacysurvey/imagine/blob/master/many-cutouts.py
Original author: Dustin Lang (dstndstn)

Usage on NERSC
--------------
shifter --image=dstndstn/viewer-cutouts:latest /bin/bash
/usr/bin/python3 many-cutouts.py list_of_objects.fits --outdir=output_dir_path [OTHER OPTIONS]
"""

import sys

# https://github.com/legacysurvey/imagine/blob/main/docker/cutouts/cutout#L4
sys.path = [
  "/usr/lib/python3.8",
  "/usr/lib/python3.8/lib-dynload",
  "/usr/local/lib/python3.8/dist-packages",
  "/usr/local/lib/python",
  "/app/src/legacypipe/py",
  "/app/imagine",
]

import os
import argparse

import numpy as np
from astropy.table import Table

from map.views import get_layer, NoOverlapError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('table', help='Table of cutouts to produce')
    parser.add_argument('--outdir', default=os.curdir, help='Directory prefix to add to output filenames')
    parser.add_argument('--pixscale', type=float, default=0.262, help='Pixel scale (arcsec/pix)')
    parser.add_argument('--size', type=int, default=224, help='Pixel size of output')
    parser.add_argument('--width', type=int, default=None, help='Pixel width of output')
    parser.add_argument('--height', type=int, default=None, help='Pixel height of output')
    parser.add_argument('--bands', default='grz', help='Bands to select for output')
    parser.add_argument('--layer', default='ls-dr9', help='Map layer to render')
    parser.add_argument('--force', default=False, help='Overwrite existing output file?  Default is to quit.')
    parser.add_argument('--stop-file', default='STOP', help='Filename in outdir that stops the process if exist')
    opt = parser.parse_args()

    pixscale = opt.pixscale
    H = W = opt.size
    if opt.height is not None:
        H = opt.height
    if opt.width is not None:
        W = opt.width
    bands = opt.bands
    layer = get_layer(opt.layer)

    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)
    stop_file = os.path.join(outdir, opt.stop_file)

    T = Table.read(opt.table)
    if "out" not in T.colnames:
        if 'OBJID' in T.colnames:
            T["out"] = np.char.mod("%d.jpg", T["OBJID"])
        else:
            T["out"] = np.char.add(np.char.mod("%.7f_", T["RA"]), np.char.mod("%.7f.jpg", T["DEC"]))

    for t in T:
        if os.path.exists(stop_file):
            break
        out = os.path.join(outdir, t['out'])
        ra = t['RA']
        dec = t['DEC']

        if os.path.exists(out) and not opt.force:
            print('Exists:', out)
            continue

        tempfiles = []
        try:
            layer.write_cutout(ra, dec, pixscale, W, H, out,
                               bands=bands, fits=False, jpeg=True, tempfiles=tempfiles)
        except NoOverlapError:
            print('No overlap with {} ({}, {})'.format(out, ra, dec))
        except Exception as e:
            print('Error for {} ({}, {}): {}'.format(out, ra, dec, e))
        finally:
            for fn in tempfiles:
                try:
                    os.unlink(fn)
                except:
                    pass

if __name__ == '__main__':
    main()

