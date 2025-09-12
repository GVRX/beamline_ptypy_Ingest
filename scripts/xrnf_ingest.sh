# scripts/xrnf_ingest.sh
python -m beamline_ptypy_ingest.ingest_tiff   --pattern "$1" --sidecar "$2" --output "$3"   --rotate 180 --flip none --roi 1024,1024 --grouping mean