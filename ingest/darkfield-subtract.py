#!/usr/bin/env python3
"""
darkfield-subtract.py

Reproduce an HDF5 file, replacing /scan/scan_data/ptycho__image by a processed stack:
  1) subtract a dark image
     - from a separate HDF5 stack (--dark-h5), averaged over N frames, OR
     - from previous options (--dark): npy/image/HDF5 path, or median:N of input
  2) threshold: values < T -> 0
  3) zero-pad each frame to S x S

Usage (dark stack in second HDF5):
  python hdf5_diffraction_process.py \
    --input in.h5 --output out.h5 \
    --ptycho-path /scan/scan_data/ptycho__image \
    --dark-h5 darks.h5 \
    --dark-h5-path /scan/scan_data/ptycho__image \
    --threshold 5 --size 2400 --center-pad

Alternative dark sources (mutually exclusive with --dark-h5):
  --dark median:16
  --dark dark.npy
  --dark dark.h5:/group/dset
  --dark dark.tif   (requires imageio)

"""

from __future__ import annotations
import argparse
import os
import re
from typing import Tuple, Optional

import h5py
import numpy as np

try:
    import imageio.v3 as iio  # optional (for TIFF/PNG darks)
except Exception:
    iio = None


def parse_args():
    p = argparse.ArgumentParser(description="Process diffraction images in an HDF5 file.")
    p.add_argument("--input", required=True, help="Input HDF5 file")
    p.add_argument("--output", required=True, help="Output HDF5 file (will be overwritten)")
    p.add_argument("--ptycho-path", default="/scan/scan_data/ptycho__image",
                   help="Path to the diffraction stack dataset (default: /scan/scan_data/ptycho__image)")

    # Mutually exclusive dark options
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dark", help="Dark image spec: npy/image/HDF5 'file.h5:/path', or 'median:N'")
    g.add_argument("--dark-h5", help="HDF5 file containing a dark stack to average")

    p.add_argument("--dark-h5-path", default=None,
                   help="Dataset path inside --dark-h5 (default: same as --ptycho-path)")
    p.add_argument("--threshold", type=float, required=True, help="Threshold T (values < T -> 0 after subtraction)")
    p.add_argument("--size", type=int, required=True, help="Target padded size S (output frames are SxS)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--center-pad", action="store_true", help="Center original frame in the SxS canvas (default)")
    mode.add_argument("--pad-origin", action="store_true", help="Place original frame at top-left (0,0) in the SxS canvas")
    return p.parse_args()


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def clone_tree_except(in_grp: h5py.Group, out_grp: h5py.Group, skip_path: str):
    """Clone groups/datasets/attributes, skipping dataset at absolute path 'skip_path'."""
    copy_attrs(in_grp, out_grp)
    for name, obj in in_grp.items():
        if isinstance(obj, h5py.Group):
            new_grp = out_grp.create_group(name)
            clone_tree_except(obj, new_grp, skip_path)
        elif isinstance(obj, h5py.Dataset):
            if obj.name == skip_path:
                continue
            kwargs = {}
            if obj.chunks is not None:
                kwargs["chunks"] = obj.chunks
            if obj.compression is not None:
                kwargs["compression"] = obj.compression
            if obj.shuffle is not None:
                kwargs["shuffle"] = obj.shuffle
            if obj.fletcher32:
                kwargs["fletcher32"] = True
            dst = out_grp.create_dataset(name, shape=obj.shape, dtype=obj.dtype, **kwargs)
            copy_attrs(obj, dst)
            if obj.ndim <= 2:
                dst[...] = obj[...]
            else:
                for i in range(obj.shape[0]):
                    dst[i, ...] = obj[i, ...]


def load_dark_from_spec(dark_spec: str, in_file: h5py.File, ptycho_path: str, sample_shape: Tuple[int, int]) -> np.ndarray:
    """Previous dark loader: npy/image/HDF5 path or median:N."""
    H, W = sample_shape
    m = re.fullmatch(r"median:(\d+)", dark_spec.strip(), flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        dset = in_file[ptycho_path]
        n = max(1, min(n, dset.shape[0]))
        stack = dset[:n, ...].astype(np.float32, copy=False)
        dark = np.median(stack, axis=0).astype(np.float32, copy=False)
        if dark.shape != (H, W):
            raise ValueError(f"Computed dark has shape {dark.shape}, expected {(H,W)}")
        return dark

    if ".h5:" in dark_spec or ".hdf5:" in dark_spec:
        fpath, dpath = dark_spec.split(":", 1)
        with h5py.File(fpath, "r") as f:
            arr = f[dpath][...]
    elif dark_spec.lower().endswith(".npy"):
        arr = np.load(dark_spec)
    elif dark_spec.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        if iio is None:
            raise RuntimeError("imageio is required to read image files; please 'pip install imageio'.")
        arr = iio.imread(dark_spec)
    else:
        raise ValueError(f"Unrecognized dark spec: {dark_spec}")

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr.mean(axis=0)
    if arr.shape != (H, W):
        raise ValueError(f"Dark shape {arr.shape} does not match expected {(H,W)}")
    return arr.astype(np.float32, copy=False)


def load_dark_from_h5_stack(dark_h5_path: str, dark_dset_path: str, sample_shape: Tuple[int, int]) -> np.ndarray:
    """
    Load dark from an HDF5 file containing a 3D stack (N,H,W) at dark_dset_path.
    Returns the mean image over N as float32 (shape HxW). Streams frame-by-frame.
    """
    H, W = sample_shape
    with h5py.File(dark_h5_path, "r") as f:
        if dark_dset_path not in f:
            raise FileNotFoundError(f"Dataset not found in dark file: {dark_dset_path}")
        dset = f[dark_dset_path]
        if dset.ndim != 3:
            raise ValueError(f"Dark dataset must be 3D (N,H,W); got shape {dset.shape}")
        N, h, w = dset.shape
        if (h, w) != (H, W):
            raise ValueError(f"Dark frame size {h}x{w} does not match sample {H}x{W}")

        acc = np.zeros((H, W), dtype=np.float64)
        for i in range(N):
            acc += dset[i, ...].astype(np.float64, copy=False)
        dark_mean = (acc / float(N)).astype(np.float32)
        return dark_mean


def place_into_canvas(frame: np.ndarray, S: int, centered: bool = True) -> np.ndarray:
    H, W = frame.shape
    out = np.zeros((S, S), dtype=np.uint16)
    if centered:
        y0 = (S - H) // 2
        x0 = (S - W) // 2
    else:
        y0 = 0
        x0 = 0
    out[y0:y0+H, x0:x0+W] = frame
    return out


def maybe_update_dim_attrs(src_ds: h5py.Dataset, dst_ds: h5py.Dataset, new_shape: Tuple[int, int, int]):
    if "total_dims" in dst_ds.attrs:
        td = dst_ds.attrs["total_dims"]
        try:
            if isinstance(td, (bytes, np.bytes_)):
                td = td.decode("ascii", errors="ignore")
            if isinstance(td, str):
                if "," in td:
                    sep = ","
                elif "x" in td or "X" in td:
                    sep = "x"
                else:
                    sep = None
                if sep:
                    dst_ds.attrs.modify("total_dims", f"{new_shape[0]}{sep}{new_shape[1]}{sep}{new_shape[2]}")
        except Exception:
            pass


def main():
    args = parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("Refusing to overwrite input file; specify a different --output path.")

    S = int(args.size)
    if S <= 0:
        raise SystemExit("--size must be positive")

    center = not args.pad_origin  # default True unless --pad-origin

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "w") as fout:
        clone_tree_except(fin["/"], fout["/"], skip_path=args.ptycho_path)

        if args.ptycho_path not in fin:
            raise SystemExit(f"Dataset not found: {args.ptycho_path}")

        src = fin[args.ptycho_path]
        if src.ndim != 3:
            raise SystemExit(f"Expected 3D stack at {args.ptycho_path}, got shape {src.shape}")

        N, H, W = src.shape
        if H > S or W > S:
            raise SystemExit(f"--size S={S} must be >= original frame size ({H}x{W})")

        # ---- Load/compute dark (float32) ----
        if args.dark_h5:
            dark_path = args.dark_h5_path or args.ptycho_path
            dark = load_dark_from_h5_stack(args.dark_h5, dark_path, (H, W))
        else:
            dark = load_dark_from_spec(args.dark, fin, args.ptycho_path, (H, W))

        # ---- Create destination dataset ----
        parent_path = os.path.dirname(args.ptycho_path.rstrip("/"))
        name = os.path.basename(args.ptycho_path.rstrip("/"))
        parent_grp = fout[parent_path]

        kwargs = {}
        chunks = src.chunks if src.chunks is not None else (1, min(S, 512), min(S, 512))
        kwargs["chunks"] = chunks
        if src.compression is not None:
            kwargs["compression"] = src.compression
        if src.shuffle is not None:
            kwargs["shuffle"] = src.shuffle
        if src.fletcher32:
            kwargs["fletcher32"] = True

        dst = parent_grp.create_dataset(name, shape=(N, S, S), dtype=np.uint16, **kwargs)

        # Copy attrs and update dimension-like fields
        copy_attrs(src, dst)
        maybe_update_dim_attrs(src, dst, (N, S, S))

        # ---- Process frames ----
        T = float(args.threshold)
        for i in range(N):
            frame_u16 = src[i, ...]  # uint16
            f = frame_u16.astype(np.float32, copy=False) - dark
            f[f < T] = 0.0
            np.clip(f, 0.0, 65535.0, out=f)
            f_u16 = f.astype(np.uint16, copy=False)
            out = place_into_canvas(f_u16, S=S, centered=center)
            dst[i, ...] = out

        fout.flush()

    print(f"Done. Wrote processed dataset to {args.output}")


if __name__ == "__main__":
    main()
