#!/usr/bin/env python3
import argparse, csv, h5py, numpy as np
def main():
    ap = argparse.ArgumentParser(description='Build sidecar HDF5 from EPICS CSV')
    ap.add_argument('--frames', required=True); ap.add_argument('--csv', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--col-posx', default='posx_m'); ap.add_argument('--col-posy', default='posy_m')
    ap.add_argument('--col-energy', default='energy_eV'); ap.add_argument('--col-dist', default='det_dist_m')
    ap.add_argument('--col-pixel', default='pixel_m'); ap.add_argument('--col-exposure', default='exposure_s')
    args = ap.parse_args()
    rows = []
    with open(args.csv, 'r', newline='') as f: rows = list(csv.DictReader(f))
    N = len(rows); pos = np.zeros((N,2), dtype=np.float64); energy = np.full((N,), np.nan); exposure = np.full((N,), np.nan)
    for i, r in enumerate(rows):
        pos[i,1] = float(r.get(args.col-posx if False else 'posx_m', r.get(args.col_posx, 'nan')))
        pos[i,0] = float(r.get(args.col-posy if False else 'posy_m', r.get(args.col_posy, 'nan')))
        if args.col_energy in r and r[args.col_energy] != '': energy[i] = float(r[args.col_energy])
        if args.col_exposure in r and r[args.col_exposure] != '': exposure[i] = float(r[args.col_exposure])
    def first_valid(key):
        for r in rows:
            v = r.get(key, '')
            if v != '': return float(v)
        return np.nan
    dist = first_valid(args.col_dist); pixel = first_valid(args.col_pixel)
    with h5py.File(args.output, 'w') as h5:
        e = h5.create_group('/entry'); e.create_dataset('pos', data=pos)
        if np.isfinite(energy).any(): e.create_dataset('energy_eV', data=energy)
        if np.isfinite(exposure).any(): e.create_dataset('exposure_s', data=exposure)
        if np.isfinite(dist): e.create_dataset('det_dist_m', data=dist)
        if np.isfinite(pixel): e.create_dataset('pixel_m', data=pixel)
        e.attrs['frames_pattern'] = args.frames
    print(f'Wrote sidecar {args.output} for {N} frames')
if __name__ == '__main__': main()
