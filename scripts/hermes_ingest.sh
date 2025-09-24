# scripts/hermes_ingest.sh
python -m beamline_ptypy_ingest.ingest_hdf5   --input "$1" --output "$2"   --roi 1024,1024 --roi-center 0.5,0.5 --grouping sum --frames-per-pos 3