import argparse, h5py, numpy as np
from .preprocess import (orient, apply_dark_flat, build_mask, crop_to_roi,
                         group_frames, select_by_position, mean_center_of_mass)
def load_dset(h5, path, default=None):
    return np.array(h5[path]) if path in h5 else default
def parse_center(s):
    if s is None: return None
    cy, cx = [float(v) for v in s.split(',')]; return (cy, cx)
def parse_size(s):
    if s is None: return None
    sy, sx = [float(v) for v in s.split(',')]; return (sy, sx)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--data', default='/entry/data'); ap.add_argument('--pos', default='/entry/pos')
    ap.add_argument('--energy', default='/entry/energy_eV'); ap.add_argument('--distance', default='/entry/det_dist_m')
    ap.add_argument('--pixel', default='/entry/pixel_m'); ap.add_argument('--exposure', default='/entry/exposure_s')
    ap.add_argument('--mask', default='/entry/mask'); ap.add_argument('--dark', default='/entry/dark'); ap.add_argument('--flat', default='/entry/flat')
    ap.add_argument('--sat', type=int, default=None)
    ap.add_argument('--roi', default='auto'); ap.add_argument('--roi-center', default=None)
    ap.add_argument('--rotate', type=int, default=0, choices=[0,90,180,270]); ap.add_argument('--flip', default='none', choices=['none','h','v','hv','vh'])
    ap.add_argument('--select-shape', default='none', choices=['none','circle','rect']); ap.add_argument('--select-center-pos', default=None)
    ap.add_argument('--select-radius', type=float, default=None); ap.add_argument('--select-size', default=None)
    ap.add_argument('--estimate-center', action='store_true'); ap.add_argument('--write-center', action='store_true')
    ap.add_argument('--grouping', default='sum', choices=['sum','mean','first','best','none']); ap.add_argument('--frames-per-pos', type=int, default=1)
    args = ap.parse_args()
    with h5py.File(args.input, 'r') as h5:
        data = load_dset(h5, args.data); pos = load_dset(h5, args.pos)
        energy = load_dset(h5, args.energy); dist = load_dset(h5, args.distance); pix = load_dset(h5, args.pixel)
        exposure = load_dset(h5, args.exposure); mask = load_dset(h5, args.mask); dark = load_dset(h5, args.dark); flat = load_dset(h5, args.flat)
    data = orient(data, rotate=args.rotate, flip=args.flip)
    if dark is not None: dark = orient(dark, rotate=args.rotate, flip=args.flip)
    if flat is not None: flat = orient(flat, rotate=args.rotate, flip=args.flip)
    if mask is not None: mask = orient(mask, rotate=args.rotate, flip=args.flip)
    if data.dtype != np.float32: data = data.astype(np.float32)
    if dark is not None or flat is not None: data = apply_dark_flat(data, dark, flat)
    sel_center = parse_center(args.select_center_pos); sel_size = parse_size(args.select_size)
    if pos is not None:
        keep = select_by_position(pos, shape=args.select_shape, center=sel_center, radius=args.select_radius, size=sel_size)
        if keep is not None and keep.dtype==np.bool_ and keep.shape[0]==data.shape[0]:
            data = data[keep]; pos = pos[keep] if pos is not None else None
            if exposure is not None and np.ndim(exposure)==1 and exposure.shape[0]==keep.shape[0]: exposure = exposure[keep]
            if energy is not None and np.ndim(energy)==1 and energy.shape[0]==keep.shape[0]: energy = energy[keep]
    est_center = None
    if args.estimate_center:
        est_center = mean_center_of_mass(data)
        print('Estimated mean center (cy,cx) [pixels, pre-ROI, post-orientation]:', est_center)
    if mask is None and args.sat is not None: mask = build_mask(data, sat_level=args.sat).astype('uint8')
    center = parse_center(args.roi_center)
    if args.roi == 'auto':
        ny, nx = data.shape[-2:]; target = (min(ny, 1024), min(nx, 1024)) if max(ny, nx) > 1024 else None
    else: sy, sx = [int(v) for v in args.roi.split(',')]; target = (sy, sx)
    data, _roi = crop_to_roi(data, size=target, center=center)
    if mask is not None: mask, _ = crop_to_roi(mask, size=target, center=center)
    if args.frames_per_pos > 1 and args.grouping != 'none':
        assert data.shape[0] % args.frames_per_pos == 0, 'N not divisible by frames-per-pos'
        grouped = []; N = data.shape[0]
        for i in range(0, N, args.frames_per_pos):
            block = data[i:i+args.frames_per_pos]; grouped.append(group_frames(block, method=args.grouping, sat_level=args.sat))
        data = np.stack(grouped, axis=0)
        if pos is not None and len(pos) == (len(grouped) * args.frames_per_pos): pos = pos[::args.frames_per_pos]
        if exposure is not None and np.ndim(exposure) == 1 and len(exposure) == (len(grouped) * args.frames_per_pos):
            if args.grouping == 'sum': exposure = exposure.reshape(-1, args.frames_per_pos).sum(axis=1)
            elif args.grouping == 'mean': exposure = exposure.reshape(-1, args.frames_per_pos).mean(axis=1)
            else: exposure = exposure[::args.frames_per_pos]
    with h5py.File(args.output, 'w') as oh:
        e = oh.create_group('/entry'); e.create_dataset('data', data=data, compression='gzip', compression_opts=4)
        if pos is not None: e.create_dataset('pos', data=pos)
        if energy is not None: e.create_dataset('energy_eV', data=np.array(energy))
        if dist is not None: e.create_dataset('det_dist_m', data=float(np.array(dist)))
        if pix is not None: e.create_dataset('pixel_m', data=float(np.array(pix)))
        if exposure is not None: e.create_dataset('exposure_s', data=np.array(exposure))
        if mask is not None: e.create_dataset('mask', data=mask.astype('uint8'))
        if est_center is not None and args.write_center: e.create_dataset('center_px', data=np.array(est_center, dtype=np.float32))
    print('Wrote %s with shape %r' % (args.output, tuple(data.shape)))
if __name__ == '__main__':
    main()
