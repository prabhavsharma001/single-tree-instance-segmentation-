#!/bin/bash
ODM_DIR="/home/cdac/Desktop/detectree2/data/odm_projects"
PLOTS="plot_B01 plot_B02 plot_B04 plot_B05 plot_B06 plot_B07 plot_B08 plot_B09 plot_B10"

for plot in $PLOTS; do
    echo "========================================"
    echo "Processing $plot..."
    echo "========================================"
    docker run -ti --rm \
        -v "$ODM_DIR/${plot}_band3:/datasets/code" \
        opendronemap/odm \
        --project-path /datasets \
        --orthophoto-resolution 5 \
        --feature-quality high \
        --min-num-features 8000 \
        --mesh-size 200000 \
        --skip-3dmodel
    sudo chown -R cdac:cdac "$ODM_DIR/${plot}_band3"
    echo "$plot done at $(date)"
done
echo "All plots processed!"
