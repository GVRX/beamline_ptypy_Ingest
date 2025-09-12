import argparse, ptypy, ptypy.utils as u, h5py
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--engine', default='DM'); ap.add_argument('--iters', type=int, default=200); args = ap.parse_args()
ptypy.load_ptyscan_module("hdf5_loader")
p = u.Param(); p.verbose_level = "interactive"
p.io = u.Param(); p.io.autosave = u.Param(active=False); p.io.interaction = u.Param(active=False)
p.scans = u.Param(); p.scans.s1 = u.Param(); p.scans.s1.name='Full'
p.scans.s1.data = u.Param(); p.scans.s1.data.name='Hdf5Loader'
p.scans.s1.data.intensities = u.Param(file=args.data, key="/entry/data")
with h5py.File(args.data,"a") as h5:
    e = h5["/entry"]
    if "pos" in e and ("posx_m" not in e or "posy_m" not in e):
        pos = e["pos"][()]
        if "posx_m" not in e: e.create_dataset("posx_m", data=pos[:,0])
        if "posy_m" not in e: e.create_dataset("posy_m", data=pos[:,1])
p.scans.s1.data.positions = u.Param(file=args.data, slow_key="/entry/posy_m", fast_key="/entry/posx_m")
p.scans.s1.data.recorded_energy   = u.Param(file=args.data, key="/entry/energy_eV", multiplier=1e-3)
p.scans.s1.data.recorded_distance = u.Param(file=args.data, key="/entry/det_dist_m")
p.scans.s1.data.recorded_psize    = u.Param(file=args.data, key="/entry/pixel_m")
p.engines = u.Param(); p.engines.e = u.Param(name=args.engine, numiter=args.iters)
P = ptypy.core.Ptycho(p, level=5)
