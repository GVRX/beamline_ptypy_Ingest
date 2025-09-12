import argparse, glob, h5py, numpy as np, os
from tifffile import imread
from .preprocess import (orient, apply_dark_flat, build_mask, crop_to_roi,
                         group_frames, select_by_position, mean_center_of_mass)

def read_sidecar(h5path: str):
    with h5py.File(h5path, 'r') as h5:
        g = h5['/entry']
        def get(k):
            return np.array(g[k]) if k in g else None
        pos = get('pos')
        energy = get('energy_eV')
        dist = get('det_dist_m')
        pix = get('pixel_m')
        exposure = get('exposure_s')
        dark = get('dark')
        flat = get('flat')
        mask = get('mask')
    return pos, energy, dist, pix, exposure, dark, flat, mask

def parse_center(s):
    if s is None:
        return None
    cy, cx = [float(v) for v in s.split(',')]
    return (cy, cx)

def parse_size(s):
    if s is None:
        return None
    sy, sx = [float(v) for v in s.split(',')]
    return (sy, sx)

def load_file_list(patterns, listfile):
    files = []
    # patterns can be None or list of globs
    if patterns:
        for p in patterns:
            files.extend(glob.glob(p))
    # listfile: plain text with one path per line (allow comments & blanks)
    if listfile:
        with open(listfile, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                files.append(s)
    # de-dup & sort by natural path order
    uniq = sorted(set(os.path.abspath(x) for x in files))
    return uniq

def main():
    ap = argparse.ArgumentParser(description='Ingest TIFF(s) + sidecar -> standardized CXI-like HDF5 for PtyPy')
    ap.add_argument('--pattern', action='append', help='Glob for TIFF files; can repeat, e.g. --pattern "scan*_a.tif" --pattern "scan*_b.tif"')
    ap.add_argument('--list', dest='listfile', help='Text file with one TIFF path per line (comments with # allowed)')
    ap.add_argument('--sidecar', required=True, help='Sidecar HDF5 with positions & metadata')
    ap.add_argument('--output', required=True, help='Output standardized HDF5')

    ap.add_argument('--grouping', default='sum', choices=['sum','mean','first','best','none'])
    ap.add_argument('--frames-per-pos', type=int, default=1)
    ap.add_argument('--sat', type=int, default=None, help='Saturation level')
    ap.add_argument('--roi', default='auto', help="'auto' or 'ny,nx'")
    ap.add_argument('--roi-center', default=None, help="'cy,cx' in pixels, or fractions in [0,1) e.g. 0.5,0.5")
    ap.add_argument('--rotate', type=int, default=0, choices=[0,90,180,270], help='Rotate images by 0/90/180/270')
    ap.add_argument('--flip', default='none', choices=['none','h','v','hv','vh'], help='Flip images horizontally/vertically')

    ap.add_argument('--select-shape', default='none', choices=['none','circle','rect'], help='Subset selection shape in scan space')
    ap.add_argument('--select-center-pos', default=None, help="Selection centre in scan space (metres) as 'cy,cx'")
    ap.add_argument('--select-radius', type=float, default=None, help='Selection radius (metres) for circle')
    ap.add_argument('--select-size', default=None, help="Selection size (metres) for rect as 'dy,dx'")

    ap.add_argument('--estimate-center', action='store_true', help='Estimate intensity-weighted mean (cy,cx) over selected stack (pre-ROI)')
    ap.add_argument('--write-center', action='store_true', help='Write the estimated center to /entry/center_px in the output HDF5')

    args = ap.parse_args()

    files = load_file_list(args.pattern, args.listfile)
    if not files:
        raise SystemExit('No input TIFFs: provide --pattern (can repeat) and/or --list file.txt')

    stack = [imread(f) for f in files]
    data = np.stack(stack, axis=0)

    pos, energy, dist, pix, exposure, dark, flat, mask = read_sidecar(args.sidecar)

    data = orient(data, rotate=args.rotate, flip=args.flip)
    if dark is not None: dark = orient(dark, rotate=args.rotate, flip=args.flip)
    if flat is not None: flat = orient(flat, rotate=args.rotate, flip=args.flip)
    if mask is not None: mask = orient(mask, rotate=args.rotate, flip=args.flip)

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if dark is not None or flat is not None:
        data = apply_dark_flat(data, dark, flat)

    sel_center = parse_center(args.select_center_pos)
    sel_size = parse_size(args.select_size)
    if pos is not None:
        keep = select_by_position(pos, shape=args.select_shape, center=sel_center,
                                  radius=args.select_radius, size=sel_size)
        if keep is not None and keep.dtype==np.bool_ and keep.shape[0]==data.shape[0]:
            data = data[keep]
            pos = pos[keep] if pos is not None else None
            if exposure is not None and np.ndim(exposure)==1 and exposure.shape[0]==keep.shape[0]:
                exposure = exposure[keep]
            if energy is not None and np.ndim(energy)==1 and energy.shape[0]==keep.shape[0]:
                energy = energy[keep]

    est_center = None
    if args.estimate_center:
        est_center = mean_center_of_mass(data)
        print("Estimated mean center (cy,cx) [pixels, pre-ROI, post-orientation]:", est_center)

    if mask is None and args.sat:
        mask = build_mask(data, sat_level=args.sat).astype('uint8')

    center = parse_center(args.roi_center)
    if args.roi == 'auto':
        ny, nx = data.shape[-2:]
        target = (min(ny, 1024), min(nx, 1024)) if max(ny, nx) > 1024 else None
    else:
        sy, sx = [int(v) for v in args.roi.split(',')]
        target = (sy, sx)
    data, _roi = crop_to_roi(data, size=target, center=center)
    if mask is not None:
        mask, _ = crop_to_roi(mask, size=target, center=center)

    if args.frames_per_pos > 1 and args.grouping != 'none':
        assert data.shape[0] % args.frames_per_pos == 0, 'N not divisible by frames-per-pos'
        grouped = []
        N = data.shape[0]
        for i in range(0, N, args.frames_per_pos):
            block = data[i:i+args.frames_per_pos]
            grouped.append(group_frames(block, method=args.grouping, sat_level=args.sat))
        data = np.stack(grouped, axis=0)
        if pos is not None and len(pos) == (len(grouped) * args.frames_per_pos):
            pos = pos[::args.frames_per_pos]
        if exposure is not None and np.ndim(exposure) == 1 and len(exposure) == (len(grouped) * args.frames_per_pos):
            if args.grouping == 'sum':
                exposure = exposure.reshape(-1, args.frames_per_pos).sum(axis=1)
            elif args.grouping == 'mean':
                exposure = exposure.reshape(-1, args.frames_per_pos).mean(axis=1)
            else:
                exposure = exposure[::args.frames_per_pos]

    with h5py.File(args.output, 'w') as oh:
        e = oh.create_group('/entry')
        e.create_dataset('data', data=data.astype(np.float32), compression='gzip', compression_opts=4)
        if pos is not None:
            e.create_dataset('pos', data=pos)
        if energy is not None:
            e.create_dataset('energy_eV', data=np.array(energy))
        if dist is not None:
            e.create_dataset('det_dist_m', data=float(np.array(dist)))
        if pix is not None:
            e.create_dataset('pixel_m', data=float(np.array(pix)))
        if exposure is not None:
            e.create_dataset('exposure_s', data=np.array(exposure))
        if mask is not None:
            e.create_dataset('mask', data=mask.astype('uint8'))
        if est_center is not None and args.write_center:
            e.create_dataset('center_px', data=np.array(est_center, dtype=np.float32))
    print('Wrote %s with shape %r from %d TIFFs' % (args.output, tuple(data.shape), len(files)))

if __name__ == '__main__':
    main()
