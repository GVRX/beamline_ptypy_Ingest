#!/usr/bin/env bash
set -euo pipefail
PATTERN=${1:?Usage: $0 'PATTERN' SIDECAR_H5 OUTPUT_H5}
SIDECAR=${2:?Usage: $0 'PATTERN' SIDECAR_H5 OUTPUT_H5}
OUTPUT=${3:?Usage: $0 'PATTERN' SIDECAR_H5 OUTPUT_H5}
python -m beamline_ptypy_ingest.ingest_tiff --pattern "$PATTERN" --sidecar "$SIDECAR" --output "$OUTPUT"   --frames-per-pos 2 --grouping sum --roi 1024,1024 --roi-center 0.5,0.5   --rotate 270 --flip v --select-shape circle --select-center-pos 0.0,0.0 --select-radius 0.0007   --estimate-center --write-center
