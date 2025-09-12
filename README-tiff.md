# Beamline PtyPy Ingestion & Preprocessing Kit

(See usage examples inside; includes selection & COM flags.)


---

### TIFF list support
You can specify input frames via one or more `--pattern` globs **and/or** a text file with `--list`:

```bash
# Multiple patterns
python -m beamline_ptypy_ingest.ingest_tiff   --pattern '/data/scan001_*.tif'   --pattern '/data/scan001_extra_*.tif'   --sidecar /data/scan001_sidecar.h5   --output  /data/scan001_std.h5

# Or a file list (one path per line; blanks and lines starting with # are ignored)
python -m beamline_ptypy_ingest.ingest_tiff   --list /data/scan001_files.txt   --sidecar /data/scan001_sidecar.h5   --output /data/scan001_std.h5
```
