#!/usr/bin/env python3
# Example script to load a pre-processed HDF5 dataset and run a basic PtyPy reconstruction.
# Note: this example assumes the data is already pre-processed (dark/flat/mask applied, cropped to ROI,
# bad frames removed, grouped if needed).
# See ingest/ingest_hdf5.py for an example of how to do the pre-processing.

import argparse, ptypy, ptypy.utils as u, h5py

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--engine', default='DM')
ap.add_argument('--iters', type=int, default=200)
args = ap.parse_args()

# Set up a basic PtyPy reconstruction
ptypy.load_ptyscan_module("hdf5_loader")
p = u.Param()
p.verbose_level = "interactive"
p.io = u.Param()
p.io.autosave = u.Param(active=False)
p.io.interaction = u.Param(active=False)
p.scans = u.Param()
p.scans.s1 = u.Param()
p.scans.s1.name='Full'
p.scans.s1.data = u.Param()
p.scans.s1.data.name='Hdf5Loader'
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

# Run the reconstruction
P = ptypy.core.Ptycho(p, level=5)

# View the results
import ptypy.utils.plot_client as pc
fig = pc.figure_from_ptycho(P)