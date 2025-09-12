#!/usr/bin/env bash
set -euo pipefail
INPUT=${1:?Usage: $0 INPUT_H5 OUTPUT_H5}
OUTPUT=${2:?Usage: $0 INPUT_H5 OUTPUT_H5}
python -m beamline_ptypy_ingest.ingest_hdf5 --input "$INPUT" --output "$OUTPUT"   --frames-per-pos 1 --grouping sum --roi 1024,1024 --roi-center 0.5,0.5   --rotate 90 --flip h --select-shape rect --select-center-pos 0.0,0.0 --select-size 0.0012,0.0012   --estimate-center --write-center
