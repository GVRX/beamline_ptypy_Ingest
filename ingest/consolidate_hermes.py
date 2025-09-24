#!/usr/bin/env python3


"""
Consolidate dark average and scan positions into a NeXus/NXS ptychography file.

Usage:
# Basic (files in current directory):
python consolidate_nxs.py HERMES2025_ABC

# Specify directory:
python consolidate_nxs.py HERMES2025_ABC --dir /data/run42

# If the dark frames aren’t under /scan/scan_data/data, point to the dataset explicitly:
python consolidate_nxs.py HERMES2025_ABC --dark-dset /scan/scan_data/detector_0/data

# Overwrite existing outputs and use column-major (Fortran) scan ordering:
python consolidate_nxs.py HERMES2025_ABC --overwrite --order col

File A: HERMES2025_ABC_000001.nxs
  - Contains ptychography scan data under /scan/scan_data (e.g. /scan/scan_data/data)
  - Will be modified in place to add /entry/dark and /entry/pos datasets            
File B: HERMES2025_ABC_dark_000001.nxs
  - Contains a stack of dark frames under /scan/scan_data (e.g. /scan/scan_data/data)
  - Will be averaged to produce the dark-field image
  - not modified
File C: HERMES2025_ABC.hdf5
  - Contains scan positions under /entry1/camera/real_sample_x and real_sample_y
  - May also contain beam energy under /camera/energy or /beam/energy_eV
  - Not modified
Outputs:
  - File A is modified in place to add:
    - /entry/dark: float32 array, average dark-field image from file B
    - /entry/pos: float64 array of shape (N,2) with scan positions from file C

Notes:
  - The script assumes that the dark frames in file B are stored as a 3D array (frames, y, x)
    or in a group containing such a dataset. If multiple datasets with ndim>=3 are found, the first one
    encountered is used. You can specify a preferred dataset path with --dark-dset.
  - The scan positions in file C are assumed to be 2D arrays (ny, nx) under the specified paths.
    They are flattened into a list of (x,y) pairs in either row-major (C-style) or column-major
    (Fortran-style) order, as specified by --order.
  - If /entry/dark or /entry/pos already exist in file A, the script will raise an error unless
    --overwrite is specified.   

"""

import argparse
import sys
from pathlib import Path
import numpy as np
import h5py

SCAN_GROUP_A = "/scan/scan_data"                      # in A (not modified)
SCAN_GROUP_B = "/scan/scan_data"                      # in B (dark frames live here)
POS_X_PATH_C = "/entry1/camera/real_sample_x"         # in C
POS_Y_PATH_C = "/entry1/camera/real_sample_y"         # in C

OUT_DARK_PATH = "/entry/dark"                         # in A
OUT_POS_PATH  = "/entry/pos"                          # in A

def _open_dataset_maybe_group(f: h5py.File, path: str) -> np.ndarray:
    """
    Return a numpy array from either a dataset at `path` or, if `path` is a group,
    a child dataset named 'data'. Raises KeyError if not found.
    """
    obj = f[path]
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    if isinstance(obj, h5py.Group):
        # common NeXus convention uses a dataset called 'data' inside the group
        for candidate in ("data", "value", "values"):
            if candidate in obj and isinstance(obj[candidate], h5py.Dataset):
                return obj[candidate][()]
    raise KeyError(f"No dataset found at or under {path!r} (looked for 'data').")

def _find_dark_stack(f: h5py.File, group_path: str, prefer: str | None = None) -> np.ndarray:
    """
    Locate a stack of dark frames within group_path. If `prefer` is provided, use that dataset path.
    Otherwise, search for a dataset with ndim>=3 or an array where first axis is frames.
    """
    if prefer:
        return f[prefer][()]
    grp = f[group_path]
    # If the group itself has a 'data' dataset, try that first.
    try:
        arr = _open_dataset_maybe_group(f, group_path)
        if arr.ndim >= 3:
            return arr
    except Exception:
        pass
    # Fall back: search children for a plausible stack
    candidates = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            if isinstance(data, np.ndarray) and data.ndim >= 3:
                candidates.append((name, data))
    grp.visititems(lambda n, o: visit(n, o))
    if candidates:
        # pick the first deepest path (usually the actual data)
        candidates.sort(key=lambda t: len(t[0]))
        return candidates[0][1]
    raise RuntimeError(f"Could not find a dark-frame stack under {group_path!r}.")

def _ensure_group(file: h5py.File, path: str) -> h5py.Group:
    """Create intermediate groups as needed and return the terminal group (or file if '/')."""
    if path == "/" or path == "":
        return file["/"]
    parts = [p for p in path.strip("/").split("/") if p]
    cur = file["/"]
    for p in parts:
        if p in cur:
            if isinstance(cur[p], h5py.Group):
                cur = cur[p]
            else:
                raise RuntimeError(f"Path component {p!r} exists but is not a group.")
        else:
            cur = cur.create_group(p)
    return cur

def write_dataset(f: h5py.File, path: str, data: np.ndarray, overwrite: bool, **dset_kwargs):
    parent_path, name = str(Path(path)).rsplit("/", 1)
    if not parent_path:
        parent_path = "/"
    grp = _ensure_group(f, parent_path)
    if name in grp:
        if not overwrite:
            raise FileExistsError(f"Dataset {path} exists. Use --overwrite to replace.")
        del grp[name]
    return grp.create_dataset(name, data=data, **dset_kwargs)

def load_positions_from_C(fileC: Path, order: str = "row") -> np.ndarray:
    with h5py.File(fileC, "r") as fC:
        x = _open_dataset_maybe_group(fC, POS_X_PATH_C)
        y = _open_dataset_maybe_group(fC, POS_Y_PATH_C)
    if x.shape != y.shape:
        raise ValueError(f"Position arrays shapes differ: {x.shape} vs {y.shape}")
    # Flatten into a list of (x,y) in requested scan order
    if order not in {"row", "col"}:
        raise ValueError("--order must be 'row' or 'col'")
    if order == "row":
        xf = np.ravel(x, order="C")
        yf = np.ravel(y, order="C")
    else:
        xf = np.ravel(x, order="F")
        yf = np.ravel(y, order="F")
    pos = np.stack([xf, yf], axis=1).astype(np.float64)
    return pos

def get_energy_from_C(fileC: Path) -> float | None:
    with h5py.File(fileC, "r") as fC:
        for candidate in ("/entry1/camera/energy", "/beam/energy_eV"):
            if candidate in fC:
                ds = fC[candidate]
                if isinstance(ds, h5py.Dataset):
                    val = ds[()]
                    if np.isscalar(val):
                        return float(val)
                    elif isinstance(val, np.ndarray) and val.size == 1:
                        return float(val.flat[0])
    return None     

def average_dark_from_B(fileB: Path, prefer_dataset: str | None = None) -> np.ndarray:
    with h5py.File(fileB, "r") as fB:
        stack = _find_dark_stack(fB, SCAN_GROUP_B, prefer=prefer_dataset)
    # Average along the first axis (frame axis). Use nanmean to be robust to NaNs.
    dark = np.nanmean(stack, axis=0)
    return dark.astype(np.float32, copy=False)

def main():
    p = argparse.ArgumentParser(
        description="Consolidate dark average and scan positions into a NeXus/NXS ptychography file."
    )
    p.add_argument("exp", help="Experiment tag <Exp> (e.g. HERMES2025_ABC)")
    p.add_argument("--dir", default=".", help="Directory containing files (default: current dir)")
    p.add_argument("--dark-dset", default=None,
                   help="Optional explicit dataset path in file B to use for dark frames "
                        "(e.g. '/scan/scan_data/data').")
    p.add_argument("--order", default="row", choices=["row", "col"],
                   help="Flattening order for positions from file C: 'row' (C/order) or 'col' (F/order).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite /entry/dark and /entry/pos if present.")
    p.add_argument("--compress", default="gzip", choices=["gzip", "lzf", "none"],
                   help="Compression for outputs (default: gzip).")
    args = p.parse_args()

    base = Path(args.dir)
    fileA = base / f"{args.exp}_000001.nxs"
    fileB = base / f"{args.exp}_dark_000001.nxs"
    fileC = base / f"{args.exp}.hdf5"

    for f in (fileA, fileB, fileC):
        if not f.exists():
            sys.exit(f"ERROR: missing required file: {f}")
        else:
            print(f"Found file: {f}")

    # Compute dark average from B
    dark = average_dark_from_B(fileB, args.dark_dset)

    # Load positions from C
    pos = load_positions_from_C(fileC, order=args.order)

    energy = get_energy_from_C(fileC)
    if energy is not None:
        print(f"Found beam energy in file C: {energy:.1f} eV")

    # Write into A
    comp = None if args.compress == "none" else args.compress
    with h5py.File(fileA, "a") as fA:
        # ensure /entry exists (many NeXus files already have it)
        _ensure_group(fA, "/entry")
        ds_dark = write_dataset(
            fA, OUT_DARK_PATH, dark, overwrite=args.overwrite,
            compression=comp, dtype=dark.dtype
        )
        # Light metadata
        ds_dark.attrs["description"] = np.bytes_("Average dark-field image")
        ds_dark.attrs["axes"] = np.bytes_("y:x")

        ds_pos = write_dataset(
            fA, OUT_POS_PATH, pos, overwrite=args.overwrite,
            compression=comp, dtype=pos.dtype
        )
        ds_pos.attrs["columns"] = np.bytes_("x,y")
        ds_pos.attrs["units"] = np.bytes_("SI (as in C)")  # adjust if you need explicit units
        # Optional NeXus hints (non-essential)
        fA["/entry"].attrs.setdefault("NX_class", np.bytes_("NXentry"))

        ds_energy = write_dataset(
            fA, "/entry/energy_eV", np.array(energy, dtype=np.float32),
            overwrite=True
        ) if energy is not None else None
        if ds_energy is not None:
            ds_energy.attrs["units"] = np.bytes_("eV")
            ds_energy.attrs["long_name"] = np.bytes_("Photon energy")
            ds_energy.attrs["description"] = np.bytes_("Beam energy from file C")           



    print(f"Wrote dark average to {fileA}:{OUT_DARK_PATH} with shape {dark.shape}")
    print(f"Wrote positions to {fileA}:{OUT_POS_PATH} with shape {pos.shape}")

if __name__ == "__main__":
    main()
