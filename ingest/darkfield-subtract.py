#!/usr/bin/env python3
"""
pad_and_optional_dark.py

Clone an HDF5 file while padding a diffraction stack to SxS.
Optionally:
  - subtract the mean of a dark-stack HDF5 before padding (only if --subtract-dark),
  - and always (if provided) write a padded clone of the dark HDF5.

Behavior
--------
Main data:
  /scan/scan_data/ptycho__image (N,H,W) -> (N,S,S)
  - Always padded.
  - If --subtract-dark is given:
      1) dark_mean = average over dark stack frames (float32)
      2) per-frame: (frame - dark_mean), optional threshold (values < T -> 0), clip to [0, 65535]
      3) pad to SxS and write
  - If --subtract-dark is NOT given: just pad (ignore threshold).

Dark file (if provided):
  Cloned to --dark-output, padding its /scan/scan_data/ptycho__image to (N,S,S).
  No subtraction/threshold is applied to the dark output.

Notes
-----
--ptycho-path defaults to /scan/scan_data/ptycho__image for both files.
"""

from __future__ import annotations
import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Pad diffraction stacks; optionally subtract dark and/or write padded dark file.")
    p.add_argument("--input", required=True, help="Input HDF5 (main data)")
    p.add_argument("--output", required=True, help="Output HDF5 for processed main data (will be overwritten)")
    p.add_argument("--ptycho-path", default="/scan/scan_data/ptycho__image",
                   help="Dataset path of diffraction stack in both files (default: /scan/scan_data/ptycho__image)")
    p.add_argument("--size", type=int, required=True, help="Target padded size S (output frames are SxS)")

    # Padding placement
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--center-pad", action="store_true", help="Center original frame in the SxS canvas (default)")
    mode.add_argument("--pad-origin", action="store_true", help="Place original frame at top-left (0,0) in the SxS canvas")

    # Optional dark handling
    p.add_argument("--dark-h5", help="Dark HDF5 file containing a stack to average for subtraction (optional)")
    p.add_argument("--dark-output", help="Output HDF5 for padded dark file (required if --dark-h5 is set)")
    p.add_argument("--subtract-dark", action="store_true",
                   help="If set, subtract mean(dark) from main data before padding (requires --dark-h5).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Threshold after subtraction: values < T -> 0. Used only if --subtract-dark is set.")
    return p.parse_args()


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def clone_tree_except(in_grp: h5py.Group, out_grp: h5py.Group, skip_path: str):
    """
    Recursively clone groups/datasets/attributes from in_grp into out_grp,
    skipping the dataset at absolute path 'skip_path' (to be created later).
    """
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


def maybe_update_dim_attrs(dst_ds: h5py.Dataset, new_shape: Tuple[int, int, int]):
    """
    If an attribute like 'total_dims' holds a simple shape string, update it.
    """
    if "total_dims" not in dst_ds.attrs:
        return
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
                return
            dst_ds.attrs.modify("total_dims", f"{new_shape[0]}{sep}{new_shape[1]}{sep}{new_shape[2]}")
    except Exception:
        pass


def compute_dark_mean(dark_file: str, ptycho_path: str, sample_shape: Tuple[int, int]) -> np.ndarray:
    """
    Stream-average the dark stack (N,H,W) to a float32 mean image (H,W).
    """
    H, W = sample_shape
    with h5py.File(dark_file, "r") as f:
        if ptycho_path not in f:
            raise FileNotFoundError(f"Dark dataset not found: {ptycho_path}")
        dset = f[ptycho_path]
        if dset.ndim != 3:
            raise ValueError(f"Dark dataset must be 3D (N,H,W), got {dset.shape}")
        N, h, w = dset.shape
        if (h, w) != (H, W):
            raise ValueError(f"Dark frames {h}x{w} do not match main data {H}x{W}")
        acc = np.zeros((H, W), dtype=np.float64)
        for i in range(N):
            acc += dset[i, ...].astype(np.float64, copy=False)
        dark_mean = (acc / float(N)).astype(np.float32)
        return dark_mean


def create_padded_dataset(parent_grp: h5py.Group, name: str, N: int, S: int, like: h5py.Dataset) -> h5py.Dataset:
    kwargs = {}
    chunks = like.chunks if like.chunks is not None else (1, min(S, 512), min(S, 512))
    kwargs["chunks"] = chunks
    if like.compression is not None:
        kwargs["compression"] = like.compression
    if like.shuffle is not None:
        kwargs["shuffle"] = like.shuffle
    if like.fletcher32:
        kwargs["fletcher32"] = True
    dst = parent_grp.create_dataset(name, shape=(N, S, S), dtype=np.uint16, **kwargs)
    return dst


def process_main(input_path: str, output_path: str, ptycho_path: str, S: int,
                 center: bool, subtract_dark: bool, threshold: float | None,
                 dark_mean: np.ndarray | None):
    """
    Clone input -> output, replacing ptycho_path with padded data.
    Optionally subtract dark_mean and threshold before padding (only if subtract_dark).
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise SystemExit("Refusing to overwrite main input file; choose a different --output path.")

    with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
        # Clone entire tree except the diffraction stack
        clone_tree_except(fin["/"], fout["/"], skip_path=ptycho_path)

        if ptycho_path not in fin:
            raise SystemExit(f"Dataset not found in main file: {ptycho_path}")
        src = fin[ptycho_path]
        if src.ndim != 3:
            raise SystemExit(f"Expected 3D stack at {ptycho_path}, got shape {src.shape}")

        N, H, W = src.shape
        if H > S or W > S:
            raise SystemExit(f"--size S={S} must be >= original frame size ({H}x{W})")

        # Create destination
        parent_path = os.path.dirname(ptycho_path.rstrip("/"))
        name = os.path.basename(ptycho_path.rstrip("/"))
        parent_grp = fout[parent_path]
        dst = create_padded_dataset(parent_grp, name, N, S, like=src)

        # Copy & adjust attrs
        copy_attrs(src, dst)
        maybe_update_dim_attrs(dst, (N, S, S))

        # Process frames
        do_subtract = bool(subtract_dark and (dark_mean is not None))
        T = float(threshold) if (do_subtract and threshold is not None) else None

        for i in range(N):
            frame_u16 = src[i, ...]  # uint16
            if do_subtract:
                f = frame_u16.astype(np.float32, copy=False) - dark_mean
                if T is not None:
                    f[f < T] = 0.0
                np.clip(f, 0.0, 65535.0, out=f)
                f_u16 = f.astype(np.uint16, copy=False)
            else:
                f_u16 = frame_u16

            out = place_into_canvas(f_u16, S=S, centered=center)
            dst[i, ...] = out

        fout.flush()


def process_dark(dark_input: str, dark_output: str, ptycho_path: str, S: int, center: bool):
    """
    Clone dark_input -> dark_output and pad its ptycho_path stack to SxS (no subtraction/threshold).
    """
    if os.path.abspath(dark_input) == os.path.abspath(dark_output):
        raise SystemExit("Refusing to overwrite dark input file; choose a different --dark-output path.")

    with h5py.File(dark_input, "r") as fin, h5py.File(dark_output, "w") as fout:
        clone_tree_except(fin["/"], fout["/"], skip_path=ptycho_path)

        if ptycho_path not in fin:
            raise SystemExit(f"Dataset not found in dark file: {ptycho_path}")
        src = fin[ptycho_path]
        if src.ndim != 3:
            raise SystemExit(f"Expected 3D dark stack at {ptycho_path}, got shape {src.shape}")

        N, H, W = src.shape
        if H > S or W > S:
            raise SystemExit(f"--size S={S} must be >= dark frame size ({H}x{W})")

        parent_path = os.path.dirname(ptycho_path.rstrip("/"))
        name = os.path.basename(ptycho_path.rstrip("/"))
        parent_grp = fout[parent_path]
        dst = create_padded_dataset(parent_grp, name, N, S, like=src)

        copy_attrs(src, dst)
        maybe_update_dim_attrs(dst, (N, S, S))

        for i in range(N):
            frame_u16 = src[i, ...]
            out = place_into_canvas(frame_u16, S=S, centered=center)
            dst[i, ...] = out

        fout.flush()


def main():
    args = parse_args()

    S = int(args.size)
    if S <= 0:
        raise SystemExit("--size must be positive")
    center = not args.pad_origin  # default True unless --pad-origin

    # Validate dark options
    if args.dark_h5 and not args.dark_output:
        raise SystemExit("--dark-output is required when --dark-h5 is provided.")
    if args.subtract_dark and not args.dark_h5:
        raise SystemExit("--subtract-dark requires --dark-h5.")

    # Load dark mean only if we actually subtract
    dark_mean = None
    if args.subtract_dark:
        with h5py.File(args.input, "r") as f:
            if args.ptycho_path not in f:
                raise SystemExit(f"Dataset not found in main file: {args.ptycho_path}")
            main_ds = f[args.ptycho_path]
            if main_ds.ndim != 3:
                raise SystemExit(f"Expected 3D stack at {args.ptycho_path}, got shape {main_ds.shape}")
            _, H, W = main_ds.shape
        dark_mean = compute_dark_mean(args.dark_h5, args.ptycho_path, (H, W))

    # Process main (subtract & threshold only if --subtract-dark)
    process_main(
        input_path=args.input,
        output_path=args.output,
        ptycho_path=args.ptycho_path,
        S=S,
        center=center,
        subtract_dark=args.subtract_dark,
        threshold=args.threshold,
        dark_mean=dark_mean
    )

    # If dark provided, also produce padded dark output (always padded; never subtracted)
    if args.dark_h5:
        process_dark(
            dark_input=args.dark_h5,
            dark_output=args.dark_output,
            ptycho_path=args.ptycho_path,
            S=S,
            center=center
        )

    print("Done.")


if __name__ == "__main__":
    main()
