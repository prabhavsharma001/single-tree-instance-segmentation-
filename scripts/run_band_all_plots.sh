#!/bin/bash
BAND=$1
RAW_DIR="/home/cdac/Desktop/detectree2/data/raw"
ODM_DIR="/home/cdac/Desktop/detectree2/data/odm_projects"
PLOTS="plot_B01 plot_B02 plot_B03 plot_B04 plot_B05 plot_B06 plot_B07 plot_B08 plot_B09 plot_B10"

for plot in $PLOTS; do
    echo "=== Processing $plot Band $BAND ==="
    
    # Create images folder and copy band files
    project_dir="$ODM_DIR/${plot}_band${BAND}"
    mkdir -p "$project_dir/images"
    
    # Find and copy band files from all multi subfolders
    find "$RAW_DIR/$plot" -name "*${BAND}.TIF" | while read f; do
        cp "$f" "$project_dir/images/"
    done
    
    count=$(ls "$project_dir/images/" | wc -l)
    echo "  Copied $count images"
    
    # Run ODM
    docker run -ti --rm \
        -v "$project_dir:/datasets/code" \
        opendronemap/odm \
        --project-path /datasets \
        --orthophoto-resolution 5 \
        --feature-quality high \
        --min-num-features 8000 \
        --mesh-size 200000 \
        --skip-3dmodel

    # Immediately clean up intermediates
    sudo rm -rf "$project_dir/opensfm"
    sudo rm -rf "$project_dir/odm_filterpoints"
    sudo rm -rf "$project_dir/odm_meshing"
    sudo rm -rf "$project_dir/odm_texturing_25d"
    sudo rm -rf "$project_dir/odm_georeferencing"
    sudo rm -rf "$project_dir/odm_report"
    sudo rm -rf "$project_dir/images"
    sudo rm -f  "$project_dir/odm_orthophoto/odm_orthophoto.original.tif"
    sudo chown -R cdac:cdac "$project_dir"
    
    echo "$plot Band $BAND done. Space: $(df -h / | tail -1 | awk '{print $4}') free"
done
echo "Band $BAND complete for all plots!"
