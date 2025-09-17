# Beamline PtyPy Ingestion & Preprocessing Kit

This README explains how to turn soft–X-ray ptychography data into a **standardized CXI‑like HDF5** that PtyPy can consume—whether your raw data are:

- **HDF5** from areaDetector NDFileHDF5 (preferred), or  
- **Numbered TIFFs** with a small **HDF5 sidecar** for per‑frame metadata.

It also includes a minimal **`PtyScan` subclass** sketch if you want to stream TIFFs directly into PtyPy.

> **Multi‑beamline note:** This kit is generic. Use beamline‑specific **XML layouts** and optional **wrapper scripts** to adapt paths/NDAttributes for each facility (e.g., XRNF @ Australian Synchrotron, others...) while keeping a **single code base**.

---

## Requirements

```text
python ≥ 3.9
numpy, h5py, tifffile, scipy, numba
(ptypy optional here; install your site’s build when reconstructing)
```

Install locally:

```bash
pip install -r requirements.txt
```

---

## Directory layout

```
beamline_ptypy_ingest/
  ├─ beamline_ptypy_ingest/
  │   ├─ __init__.py
  │   ├─ preprocess.py               # dark/flat/mask/ROI/grouping + rotate/flip
  │   ├─ ingest_hdf5.py              # HDF5 → standardized HDF5
  │   ├─ ingest_tiff.py              # TIFF+sidecar → standardized HDF5
  │   ├─ ptyscan_tiff_skeleton.py    # optional direct TIFF loader for PtyPy
  │   └─ schemas.py
  ├─ examples/
  │   ├─ ptypy_recipe_hdf5.py        # trivial PtyPy params (HDF5)
  │   └─ ptypy_recipe_tiff.py        # trivial PtyPy params (TIFF+sidecar)
  ├─ xml_layouts/
  │   ├─ hermes/beamline_cxi_layout.xml       # example CXI‑like NDFileHDF5 layout (HERMES)
  │   ├─ hermes/beamline_minimal_layout.xml   # minimal NDFileHDF5 layout (HERMES)
  │   ├─ xrnf/beamline_cxi_layout.xml         # example CXI‑like layout (XRNF)
  │   └─ xrnf/beamline_minimal_layout.xml
  ├─ scripts/
  │   ├─ make_sidecar_from_epics_csv.py  # EPICS CSV → HDF5 sidecar
  │   ├─ hermes_ingest.sh                # wrapper with beamline‑specific defaults (optional)
  │   └─ xrnf_ingest.sh                  # wrapper with beamline‑specific defaults (optional)
  ├─ tests/
  │   ├─ synth_make_small_dataset.py     # tiny synthetic HDF5 builder
  │   └─ orientation_roi_smoketest.py    # rotate/flip + ROI‑center checks
  └─ README.md
```

---

## Standardized HDF5 schema (CXI‑like)

The tools here write a single HDF5 with:

```
/entry/data         float32, (N, ny, nx)   # after optional grouping/ROI/orientation
/entry/pos          float64, (N, 2)        # metres; scan X/Y per frame or per position
/entry/energy_eV    float64, () or (N,)    # eV
/entry/det_dist_m   float64, ()            # metres (detector distance)
/entry/pixel_m      float64, ()            # metres/pixel
/entry/exposure_s   float64, () or (N,)    # seconds (optional)
/entry/mask         uint8/bool, (ny, nx)   # optional (1 = bad pixel)
/entry/dark         float32, (ny, nx)      # optional
/entry/flat         float32, (ny, nx)      # optional
```

> **Units:** metres for positions & distances; metres/pixel for pitch; eV for energy; seconds for exposure.

---

## Quick starts

### A) HDF5 path (preferred)

1) Configure areaDetector NDFileHDF5 with a beamline layout, e.g.
   - `xml_layouts/hermes/beamline_cxi_layout.xml` or
   - `xml_layouts/xrnf/beamline_cxi_layout.xml`  
   so PVs land in stable dataset paths.

2) Consolidate to the standard file (showing new flags):

```bash
python -m ptypy_ingest.ingest_hdf5   --input /data/scan001.h5   --output /data/scan001_std.h5   --frames-per-pos 3   --grouping sum   --roi 1024,1024   --roi-center 0.5,0.5   --rotate 90   --flip h   --sat 65000
```

3) Reconstruct with PtyPy:

```bash
python examples/ptypy_recipe_hdf5.py --data /data/scan001_std.h5 --engine DM --iters 200
```

### B) TIFF path

1) Create a sidecar HDF5 from your EPICS CSV (one row per frame):

```bash
python scripts/make_sidecar_from_epics_csv.py   --frames '/data/scan001_*.tif'   --csv /data/epics_log.csv   --output /data/scan001_sidecar.h5
```

**Expected CSV columns (customise as needed):**
```
frame_index,posx_m,posy_m,energy_eV,det_dist_m,pixel_m,exposure_s
```

2) Consolidate TIFFs + sidecar → standardized HDF5 (showing new flags):

```bash
python -m beamline_ptypy_ingest.ingest_tiff   --pattern '/data/scan001_*.tif'   --sidecar /data/scan001_sidecar.h5   --output /data/scan001_std.h5   --frames-per-pos 2   --grouping mean   --roi 1024,1024   --roi-center 256,300   --rotate 270   --flip v
```

3) Reconstruct the standardized file as above.

---

## Preprocessing knobs

Both `ingest_hdf5.py` and `ingest_tiff.py` support:

- `--frames-per-pos N` : number of exposures per scan position  
- `--grouping {sum|mean|first|best|none}`  
  - **sum/mean** improve SNR; **first** keeps the earliest; **best** picks the highest median after saturation‑clipping.  
- `--roi ny,nx` : crop to a fixed size;  
  `--roi-center cy,cx` : choose the ROI centre (pixels **or** fractions in `[0,1)`).
- `--rotate {0,90,180,270}` : rotate diffraction images;  
  `--flip {none,h,v,hv}` : horizontal/vertical flips (applied to **data**, **dark**, **flat**, **mask**).
- `--sat LEVEL` : build a simple saturation‑aware bad‑pixel mask  
- optional dark/flat/mask if present in the input

> Tip: for large scans (hundreds–thousands of positions) prefer grouping at ingest time and keep `ny,nx` modest (e.g., 1024×1024 centred ROI).

---

## Example PtyPy params

### Standardized HDF5

```python
# examples/ptypy_recipe_hdf5.py
import argparse, ptypy, ptypy.utils as u, h5py, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--engine', default='DM'); ap.add_argument('--iters', type=int, default=200); args = ap.parse_args()

ptypy.load_ptyscan_module("hdf5_loader")  # provides 'Hdf5Loader'

p = u.Param(); p.verbose_level = "interactive"
p.io = u.Param(); p.io.autosave = u.Param(active=False); p.io.interaction = u.Param(active=False)

p.scans = u.Param(); p.scans.s1 = u.Param(); p.scans.s1.name='Full'
p.scans.s1.data = u.Param(); p.scans.s1.data.name='Hdf5Loader'
p.scans.s1.data.intensities = u.Param(file=args.data, key="/entry/data")

# Positions: create split keys if only /entry/pos exists (N,2)
with h5py.File(args.data,"a") as h5:
    e = h5["/entry"]
    if "pos" in e and ("posx_m" not in e or "posy_m" not in e):
        pos = e["pos"][()]
        if "posx_m" not in e: e.create_dataset("posx_m", data=pos[:,0])
        if "posy_m" not in e: e.create_dataset("posy_m", data=pos[:,1])

p.scans.s1.data.positions = u.Param(file=args.data, slow_key="/entry/posy_m", fast_key="/entry/posx_m")

p.scans.s1.data.recorded_energy   = u.Param(file=args.data, key="/entry/energy_eV", multiplier=1e-3)  # eV->keV
p.scans.s1.data.recorded_distance = u.Param(file=args.data, key="/entry/det_dist_m")
p.scans.s1.data.recorded_psize    = u.Param(file=args.data, key="/entry/pixel_m")

p.engines = u.Param(); p.engines.e = u.Param(name=args.engine, numiter=args.iters)

P = ptypy.core.Ptycho(p, level=5)
```

### Direct TIFF streaming (optional sketch)

If you want to bypass consolidation, adapt `beamline_ptypy_ingest/ptyscan_tiff_skeleton.py` (verify against your PtyPy version), then:

```python
# examples/ptypy_recipe_tiff.py
import ptypy, ptypy.utils as u
p = u.Param(); p.verbose_level = 3
p.scans = u.Param(); p.scans.s1 = u.Param(); p.scans.s1.name='Full'
p.scans.s1.data = u.Param(); p.scans.s1.data.name='HERMES_TIFF'  # replace with your custom source, e.g. XRNF_TIFF
p.scans.s1.data.pattern='/data/scan001_*.tif'
p.scans.s1.data.sidecar='/data/scan001_sidecar.h5'
p.engines=u.Param(); p.engines.e1=u.Param(); p.engines.e1.name='DM'; p.engines.e1.numiter=150
P = ptypy.core.Ptycho(p)
```

## Data Quality Evaluation (`ingest/evaluate.py`)

The `ingest/evaluate.py` module provides a lightweight, platform-independent quality-control step for ptychography datasets (CXI/HDF5).  
It is designed to be run **before reconstruction** to identify unusual diffraction frames, reject unstable measurements, and generate scan-space diagnostic maps.

### Features

- **Unsupervised outlier detection**
  - Combines robust feature deviation (intensity, COM shifts, radial energy distribution, speckle contrast, similarity to a reference) with PCA reconstruction error of low-dimensional embeddings.
  - Produces a per-frame **anomaly score** (0–1 normalized).

- **Reject lists**
  - Flexible modes:
    - `--reject-topk K` → reject K worst frames.
    - `--reject-topp p` → reject top *p* fraction (0–1).
    - `--reject-thresh T` → reject all with score ≥ T.
  - Outputs:
    - `accept_indices.npy`, `reject_indices.npy`, `mask_indices.npy`.

- **Scan-space masking**
  - Restrict evaluation to a **rectangular** (`--rect CX CY W H`) or **circular** (`--circ CX CY R`) region in stage coordinates.

- **Diagnostic maps**
  - **Deviation score** map (unsupervised outlier score).
  - **STXM** map (log-intensity sum per scan position).
  - **DPC-X / DPC-Y** maps (center-of-mass shifts in x,y).
  - Combined as a 4-panel `qc_maps.png`.

- **PtyPy integration**
  - `--emit-ptypy-json path.json` writes a JSON fragment with CXI paths and accepted/rejected index lists.
  - Can be consumed by the `PtyScan`/ingest layer to automatically exclude bad frames.

- **Thumbnail gallery**
  - `--thumbs-topk N` → save top-N outlier frames as PNG thumbnails.
  - Generates an `index.html` gallery (with positions and scores).

### Example Usage

```bash
python ingest/evaluate.py /path/to/data.cxi \
  --data-path /entry_1/data/data \
  --pos-x-path /entry_1/instrument/positioners/x \
  --pos-y-path /entry_1/instrument/positioners/y \
  --roi 1024 --bin 4 --embed-side 32 --pca-k 12 --w-feat 0.5 \
  --reject-topp 0.02 --rect 0 0 50 50 \
  --emit-ptypy-json qc_run/ptypy_indices.json \
  --thumbs-topk 48 --thumbs-size 256 \
  --out qc_run
Outputs
qc_maps.png — deviation/STXM/DPC maps.

outlier_score.npy — per-frame anomaly scores.

stxm.npy, dpcx.npy, dpcy.npy — scan-contrast maps.

accept_indices.npy, reject_indices.npy, mask_indices.npy — for pipeline filtering.

qc_summary.csv — tabular results (index, pos, scores, mask/reject flags).

qc_meta.json — run configuration metadata.

ptypy_indices.json (if requested) — compact JSON fragment for PtyPy ingress.

thumbs/index.html — outlier thumbnails gallery.

---

## areaDetector HDF5 XML (beamline CXI‑like)

Example layout mapping NDArray + NDAttributes to stable datasets:

```xml
<hdf5_layout>
  <group name="entry" compress="true">
    <dataset name="data"        source="NDArray"     dims="N,Y,X" type="UInt16"/>
    <dataset name="pos"         source="NDAttribute" ndattribute="POS"      dims="N,2"   type="Float64"/>
    <dataset name="energy_eV"   source="NDAttribute" ndattribute="ENERGY"   dims="N"     type="Float64"/>
    <dataset name="det_dist_m"  source="NDAttribute" ndattribute="DET_DIST" dims="Scalar" type="Float64"/>
    <dataset name="pixel_m"     source="NDAttribute" ndattribute="PIXEL"    dims="Scalar" type="Float64"/>
    <dataset name="exposure_s"  source="NDAttribute" ndattribute="EXPOSE"   dims="N"     type="Float64"/>
    <!-- Optional: -->
    <dataset name="mask" source="NDAttribute" ndattribute="MASK" dims="Y,X" type="UInt8" optional="true"/>
    <dataset name="dark" source="NDAttribute" ndattribute="DARK" dims="Y,X" type="Float32" optional="true"/>
    <dataset name="flat" source="NDAttribute" ndattribute="FLAT" dims="Y,X" type="Float32" optional="true"/>
  </group>
</hdf5_layout>
```

**Map your PVs to NDAttributes** (e.g., `POSX/POSY` → packed `POS` 2‑vector in metres; `ENERGY` in eV; `DET_DIST` in metres; `PIXEL` in metres/pixel; `EXPOSE` in seconds). Keep per‑beamline variants in `xml_layouts/<beamline>/` with consistent dataset paths so the ingestion scripts don’t change.

---

## Beamline wrappers (optional)

Create thin shell or Python wrappers to pin beamline defaults, for example:

```bash
# scripts/hermes_ingest.sh
python -m beamline_ptypy_ingest.ingest_hdf5   --input "$1" --output "$2"   --roi 1024,1024 --roi-center 0.5,0.5 --grouping sum --frames-per-pos 3
```

```bash
# scripts/xrnf_ingest.sh
python -m beamline_ptypy_ingest.ingest_tiff   --pattern "$1" --sidecar "$2" --output "$3"   --rotate 180 --flip none --roi 1024,1024 --grouping mean
```

---

## Performance tips

- Use **ROI** (e.g., 1024×1024) centred on the optical axis for speed.  
- For **9–9000 positions**, stream in blocks (`Full`/`BlockFull` in PtyPy) and pre‑group frames to limit memory.  
- Prefer **NVMe** local scratch for raw stacks and write recon outputs to a separate path to avoid I/O contention.  
- On GPU nodes, switch to the `*_cupy` engines in your PtyPy params.

---

## Troubleshooting

- **“No files match glob”**: check shell quoting and paths for `--pattern`.  
- **Dim mismatch after grouping**: ensure `N % --frames-per-pos == 0`.  
- **Units look wrong in reconstruction**: verify `/entry/pixel_m` (metres/pixel) and `/entry/det_dist_m` (metres).  
- **ROI placement**: set `--roi-center cy,cx` in pixels or fractions (e.g., `0.5,0.5`).  
- **Orientation wrong way up**: tweak `--rotate` / `--flip` until the central lobe is in the expected quadrant.

---

## Smoke test (synthetic)

```bash
python tests/synth_make_small_dataset.py  # writes synth.h5
python -m ingest.ingest_hdf5   --input synth.h5 --output synth_std.h5   --roi 512,512 --roi-center 0.5,0.5 --rotate 90 --flip v
python examples/ptypy_recipe_hdf5.py --data synth_std.h5
```

To validate orientation & ROI logic:

```bash
python tests/orientation_roi_smoketest.py
# -> OK: orientation & ROI center tests passed
```

---

## License

MIT — adapt freely for your beamline/cluster environment.
