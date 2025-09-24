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


COMMON_POS_CANDIDATES = ['/entry/pos', '/pos', '/entry/scan_pos', '/scan_pos', '/entry/positions', '/positions'] 
COMMON_POSX_CANDIDATES = ['/entry/posx_m', '/posx_m', '/entry/scan_posx_m', '/scan_posx_m', '/entry/positions_x', '/positions_x']
COMMON_POSY_CANDIDATES = ['/entry/posy_m', '/posy_m', '/entry/scan_posy_m', '/scan_posy_m', '/entry/positions_y', '/positions_y']

def auto_find_path(h, candidates):
    for p in candidates:
        if p in h: return p
    return None


def read_positions(h, posx_path, posy_path, n=None):
    """Read scan positions from HDF5 file, with some common fallbacks.  
    Returns (px, py) as float arrays of length n (or None if not found).
    """ 
    if posx_path is None: 
        posx_path = auto_find_path(h, COMMON_POSX_CANDIDATES)

    if posy_path is None: 
        posy_path = auto_find_path(h, COMMON_POSY_CANDIDATES)

    if (posx_path is None or posy_path is None) or (posx_path == posy_path ):
        for p in COMMON_POS_CANDIDATES:
            if p in h:
                ds = h[p][...]
                if n is None: 
                    n = ds.shape[0]
                if ds.shape[0] >= n and ds.shape[1] >= 2:
                        return ds[:n,0].astype(float), ds[:n,1].astype(float)
        raise KeyError("Could not find scan position datasets. Provide --pos-x-path and --pos-y-path.")
    
    else:
        print(f"Reading positions from {posx_path} and {posy_path}")
        x = np.asarray(h[posx_path][...], dtype=np.float64)#.reshape(-1)[:n]
        y = np.asarray(h[posy_path][...], dtype=np.float64)#.reshape(-1)[:n]

    return x, y


def ingest(args):
    
    rotate   = args["rotate"]
    flip     =  args["flip"]
    selectCenterPos = args["select_center_pos"]
    selectSize = args["select_size"]
    selectShape = args["select_shape"]
    selectRadius = args["select_radius"]
    estimateCenter = args["estimate_center"]
    saturationLimit = args["sat"]
    roiType = args["roi"]
    roiCenter = args["roi_center"]
    framesPerPos = args["frames_per_pos"]
    groupFrames = args["grouping"]
    outPath = args["output"]
    writeCenter = args["write_center"]
    overrideDetDistance = args["overrideDetDistance"]
    overridePixelSize  = args["overridePixelSize"]  

    with h5py.File(args["input"], 'r') as h5:

        print("Loading data from %s" % args["data"])
        data     = load_dset(h5, args["data"]); 
        print(f'Loaded data from {args["data"]}, shape: {data.shape if data is not None else None}')

        if data.dtype != np.uint32: 
            print(f'Warning: data dtype is {data.dtype}, casting to uint32')
            data = data.astype(np.uint32)
       

        #pos      = load_dset(h5, args["pos"])
        energy   = load_dset(h5, args["energy"]); 
        energy  = energy.astype(float) if energy is not None else None 
        dist     = load_dset(h5, args["distance"]); 
        pix      = load_dset(h5, args["pixel"])
        

        #px,py = read_positions(h5, args['pos_x_path'], args['pos_y_path'], data.shape[0]); 
        px,py = read_positions(h5, args['pos'], args['pos'], data.shape[0]);
        pos = np.stack([px, py], axis=1) if (px is not None and py is not None) else None
        print(f'Loaded positions from {args["pos_x_path"]} and {args["pos_y_path"]}, shape: {pos.shape if pos is not None else None}')
        print(f'Positions x range: {np.min(px)} to {np.max(px)} m, y range: {np.min(py)} to {np.max(py)} m')
        # convert to meters:
        if pos is not None and np.max(np.abs(pos)) > 100: 
            print('Assuming positions are in um, converting to meters')
            pos *= 1e-6
    

    
        try:
            exposure = load_dset(h5, args["exposure"]); 
        except:
            exposure = None
            print(f'Warning: could not load exposure from {args["exposure"]}')  
        try:
            mask     = load_dset(h5, args["mask"]); 
        except:
            mask = None
            print(f'Warning: could not load mask from {args["mask"]}')  
        try:
            dark     = load_dset(h5, args["dark"]); 
        except:
            dark = None
            print(f'Warning: could not load dark from {args["dark"]}')  
        try:    
            flat     = load_dset(h5, args["flat"])
        except:
            flat = None
            print(f'Warning: could not load flat from {args["flat"]}')  

    assert data is not None, 'No data loaded'

        
    
    data = orient(data, rotate=rotate, flip=flip)

    if dark is not None: 
        dark = orient(dark, rotate=rotate, flip=flip)

    if flat is not None: 
        flat = orient(flat, rotate=rotate, flip=flip)

    if mask is not None: 
        mask = orient(mask, rotate=rotate, flip=flip)

   
    if dark is not None or flat is not None: 
        data = apply_dark_flat(data, dark, flat)

    sel_center = parse_center(selectCenterPos); 
    sel_size = parse_size(selectSize)

    if pos is not None:
        keep = select_by_position(pos, 
                                  shape=selectShape, 
                                  center=sel_center, 
                                  radius=selectRadius, 
                                  size=sel_size)
        
        if keep is not None and keep.dtype==np.bool_ and keep.shape[0]==data.shape[0]:
            data = data[keep]; 
            pos = pos[keep] if pos is not None else None

            if exposure is not None and np.ndim(exposure)==1 and exposure.shape[0]==keep.shape[0]: 
                exposure = exposure[keep]

            if energy is not None and np.ndim(energy)==1 and energy.shape[0]==keep.shape[0]: 
                energy = energy[keep]
    
    est_center = None

    if estimateCenter:
        est_center, meanData = mean_center_of_mass(data)
        print('Estimated mean center (cy,cx) [pixels, pre-ROI, post-orientation]:', est_center)
        print(np.shape(meanData))
    
    if mask is None and saturationLimit is not None: 
        mask = build_mask(data, sat_level=saturationLimit).astype('uint8')
    
    center = parse_center(roiCenter)
    
    if roiType == 'auto':
        ny, nx = data.shape[-2:]; 
        target = (min(ny, 1024), min(nx, 1024)) if max(ny, nx) > 1024 else None
    else: 
        sy, sx = [int(v) for v in roiType.split(',')]; 
        target = (sy, sx)
    
    data, _roi = crop_to_roi(data, size=target, center=center)

    if mask is not None: 
        mask, _ = crop_to_roi(mask, size=target, center=center)
    
    if framesPerPos > 1 and groupFrames != 'none':
        assert data.shape[0] % framesPerPos == 0, 'N not divisible by frames-per-pos'
        grouped = []; 
        N = data.shape[0]
        
        for i in range(0, N, framesPerPos):
            block = data[i:i+framesPerPos]
            grouped.append(group_frames(block, method=grouping, sat_level=saturationLimit))
        data = np.stack(grouped, axis=0)
        
        if pos is not None and len(pos) == (len(grouped) * framesPerPos): 
            pos = pos[::framesPerPos]
        
        if exposure is not None and np.ndim(exposure) == 1 and len(exposure) == (len(grouped) * framesPerPos):
            if groupFrames == 'sum': 
                exposure = exposure.reshape(-1, framesPerPos).sum(axis=1)
            elif groupFrames == 'mean': 
                exposure = exposure.reshape(-1, framesPerPos).mean(axis=1)
            else: exposure = exposure[::framesPerPos]

    with h5py.File(outPath, 'w') as oh:
        e = oh.create_group('/entry'); 
        e.create_dataset('data', data=data, compression='gzip', compression_opts=4)
       
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
        
        if est_center is not None and writeCenter: 
            e.create_dataset('center_px', data=np.array(est_center, dtype=np.float32))
        
        if estimateCenter:
             e.create_dataset('meanData', data=np.array(meanData, dtype=np.uint32))
            
        if overrideDetDistance is not None and overrideDetDistance > 0:
            print(f'Overriding detector distance to {overrideDetDistance} m')
            e.create_dataset('det_dist_m', data=float(overrideDetDistance)) 

        if overridePixelSize is not None and overridePixelSize > 0:
            print(f'Overriding pixel size to {overridePixelSize} m')
            e.create_dataset('pixel_m', data=float(overridePixelSize))  

   
    print('Wrote %s with shape %r' % (outPath, tuple(data.shape)))
 
    return data, est_center, tuple(data.shape), outPath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    #ap.add_argument('--data', default='/entry/data')
    #ap.add_argument('--pos', default='/entry/pos')
    #ap.add_argument('--energy', default='/entry/energy_eV')
    #ap.add_argument('--distance', default='/entry/det_dist_m')
    #ap.add_argument('--pixel', default='/entry/pixel_m')

    
    ap.add_argument('--pos-x-path', default='/posx_m')
    ap.add_argument('--pos-y-path', default='/posy_m')   
    ap.add_argument('--exposure', default='/entry/exposure_s')
    ap.add_argument('--mask', default='/entry/mask')
    ap.add_argument('--dark', default='/entry/dark')
    ap.add_argument('--flat', default='/entry/flat')
    ap.add_argument('--data', default='/data')
    ap.add_argument('--pos', default='/pos')
    ap.add_argument('--energy', default='/energy_ev')
    ap.add_argument('--distance', default='/det_distance_m')
    ap.add_argument('--pixel', default='/det_pixelsize_m')
    ap.add_argument('--sat', type=int, default=None)
    ap.add_argument('--roi', default='auto')
    ap.add_argument('--roi-center', default=None)
    ap.add_argument('--rotate', type=int, default=0, choices=[0,90,180,270])
    ap.add_argument('--flip', default='none', choices=['none','h','v','hv','vh'])
    ap.add_argument('--select-shape', default='none', choices=['none','circle','rect'])
    ap.add_argument('--select-center-pos', default=None)
    ap.add_argument('--select-radius', type=float, default=None); 
    ap.add_argument('--select-size', default=None)
    ap.add_argument('--estimate-center', action='store_true'); 
    ap.add_argument('--write-center', action='store_true')
    ap.add_argument('--grouping', default='sum', choices=['sum','mean','first','best','none']);
    ap.add_argument('--frames-per-pos', type=int, default=1)
    ap.add_argument("--overrideDetDistance", type=float, default="0.05"),   # in meters
    ap.add_argument("--overridePixelSize",  type=float, default=None)    # in meters
    
    args = ap.parse_args()

    _data = ingest(vars(args))


if __name__ == '__main__':
    main()
