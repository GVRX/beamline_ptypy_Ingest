#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper recipe: adds optional --ptypy-indices masking to a standard PtyPy HDF5 recipe.
Reads accept_indices from JSON (key configurable via --indices-key).

Writes a filtered *.accept.h5 next to your input, copying frame-length datasets (e.g., 
/entry/data, /entry/pos, /entry/pos_x, /entry/pos_y) down-selected to those indices; 
scalars and other groups are preserved.

Invokes  ptypy_recipe_hdf5.py as a subprocess with --data pointed at the filtered 
file (or the original if no mask given), passing through any extra args it needs.

Usage examples:
  python ptypy_recipe_hdf5_accept.py --data ./standardized_demo.h5  \
          --ptypy-indices ./masking/ptypy_accept_indices.json       \
          --engine DM --iters 200

  # Without masking (behaves like the original recipe):
  python ptypy_recipe_hdf5_accept.py --data ./standardized_demo.h5 --engine DM --iters 200
"""
import argparse, json, os, sys, subprocess, tempfile
from pathlib import Path

import h5py
import numpy as np

def filter_hdf5_by_indices(src_h5: Path, dst_h5: Path, accept_idx, verbose=True):
    """Copy a standardized HDF5, keeping only frames in accept_idx.

    Expected datasets (if present):
      /entry/data         (N, ny, nx)
      /entry/pos          (N, 2)   or split: /entry/pos_x, /entry/pos_y
      (optional scalars): /entry/energy_eV, /entry/det_dist_m, /entry/pixel_m
      (other datasets under /entry are copied; if their first dimension equals N, they are filtered)

    Returns the output path as str.
    """
    src_h5 = Path(src_h5)
    dst_h5 = Path(dst_h5)
    accept_idx = np.asarray(sorted(set(int(i) for i in accept_idx)), dtype=np.int64)
    if accept_idx.size == 0:
        raise ValueError("accept_idx is empty; nothing to keep")

    if verbose:
        print(f"[mask] Filtering {src_h5} -> {dst_h5} (keep {accept_idx.size} frames)")

    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        # Copy file attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Source entry
        s_entry = src.get("entry") or src.get("/entry")
        if s_entry is None:
            raise RuntimeError("No /entry group in source HDF5")

        # Determine N from /entry/data if available
        N = None
        if "data" in s_entry and isinstance(s_entry["data"], h5py.Dataset):
            N = int(s_entry["data"].shape[0])

        d_entry = dst.create_group("entry")

        def copy_ds(name, ds):
            arr = ds[()]
            # Filter frame-like arrays (axis-0 length equals N)
            if N is not None and arr.ndim >= 1 and arr.shape[0] == N:
                arr = arr[accept_idx]
            out = d_entry.create_dataset(name, data=arr, dtype=arr.dtype)
            for ak, av in ds.attrs.items():
                out.attrs[ak] = av

        # Preferred order first (so key datasets exist early)
        preferred = ["data", "pos", "pos_x", "pos_y",
                     "energy_eV", "det_dist_m", "pixel_m",
                     "scan_shape", "pos_grid_x", "pos_grid_y", "frame_index_grid"]
        for k in preferred:
            if k in s_entry and isinstance(s_entry[k], h5py.Dataset):
                copy_ds(k, s_entry[k])

        # Copy the rest of /entry
        for name, obj in s_entry.items():
            if name in d_entry:
                continue
            if isinstance(obj, h5py.Dataset):
                copy_ds(name, obj)
            elif isinstance(obj, h5py.Group):
                g = d_entry.create_group(name)
                for ak, av in obj.attrs.items():
                    g.attrs[ak] = av
                for dsname, dsobj in obj.items():
                    if isinstance(dsobj, h5py.Dataset):
                        sub = g.create_dataset(dsname, data=dsobj[()])
                        for ak, av in dsobj.attrs.items():
                            sub.attrs[ak] = av

        # Copy root-level groups other than /entry
        for name, obj in src.items():
            if name == "entry":
                continue
            if isinstance(obj, h5py.Dataset):
                dset = dst.create_dataset(name, data=obj[()])
                for ak, av in obj.attrs.items():
                    dset.attrs[ak] = av
            elif isinstance(obj, h5py.Group):
                g = dst.create_group(name)
                for ak, av in obj.attrs.items():
                    g.attrs[ak] = av
                for dsname, dsobj in obj.items():
                    if isinstance(dsobj, h5py.Dataset):
                        sub = g.create_dataset(dsname, data=dsobj[()])
                        for ak, av in dsobj.attrs.items():
                            sub.attrs[ak] = av
    return str(dst_h5)

def main():
    parser = argparse.ArgumentParser(description="PtyPy HDF5 recipe wrapper with optional accept-indices masking")
    parser.add_argument("--data", required=True, help="Path to standardized HDF5 (input)")
    parser.add_argument("--engine", default="DM", help="PtyPy engine name (e.g., DM)")
    parser.add_argument("--iters", type=int, default=200, help="Iteration count")
    parser.add_argument("--output", default=None, help="Optional output directory for results/logs")

    # New args for masking:
    parser.add_argument("--ptypy-indices", default=None, help="JSON file with 'accept_indices' list")
    parser.add_argument("--indices-key", default="accept_indices", help="Key name in the JSON (default: accept_indices)")

    # Allow passing extra args through to the original recipe
    parser.add_argument("--recipe", default=None, help="Path to the original recipe (defaults to sibling ptypy_recipe_hdf5.py)")
    args, unknown = parser.parse_known_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {data_path}")

    # Find the original recipe
    if args.recipe is not None:
        recipe_path = Path(args.recipe).resolve()
    else:
        # default: file named 'ptypy_recipe_hdf5.py' alongside this wrapper
        recipe_path = Path(__file__).with_name("ptypy_recipe_hdf5.py")
    if not recipe_path.exists():
        raise FileNotFoundError(f"Original recipe not found: {recipe_path} "
                                f"(use --recipe to point to it)")

    run_path = data_path
    tmp_path = None

    if args.ptypy_indices:
        with open(args.ptypy_indices, "r") as f:
            js = json.load(f)
        if args.indices_key not in js:
            raise KeyError(f"JSON {args.ptypy_indices} lacks key '{args.indices_key}'")
        accept_idx = js[args.indices_key]
        if not isinstance(accept_idx, (list, tuple)):
            raise TypeError("Indices must be a list of integers")
        # Create filtered HDF5 next to input
        stem = data_path.stem
        tmp_path = data_path.with_name(stem + ".accept.h5")
        filter_hdf5_by_indices(data_path, tmp_path, accept_idx, verbose=True)
        run_path = tmp_path

    # Build command for the original recipe
    cmd = [sys.executable, str(recipe_path),
           "--data", str(run_path),
           "--engine", str(args.engine),
           "--iters", str(args.iters)]
    if args.output:
        cmd += ["--output", str(args.output)]
    # pass through any extra switches unknown to this wrapper
    cmd += list(unknown)

    print("[wrapper] Running original recipe:")
    print(" ", " ".join(map(str, cmd)))
    # hand off to original recipe
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
