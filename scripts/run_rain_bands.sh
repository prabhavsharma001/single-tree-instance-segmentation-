#!/bin/bash

SRC_DIR="/home/cdac/Downloads/RGB and Multispectral images dataset ( Álamos, Sonora)/First Rain Season/Rain Season Multi"
BASE_DIR="/home/cdac/Desktop/detectree2/odm_projects"

BANDS=("1" "2" "3" "4" "5")
NAMES=("blue" "green" "red" "rededge" "nir")

for i in "${!BANDS[@]}"; do
    BAND="${BANDS[$i]}"
    NAME="${NAMES[$i]}"
    PROJECT_DIR="$BASE_DIR/rain_band${BAND}"

    echo "================================================"
    echo "Processing Rain Season Band $BAND ($NAME)..."
    echo "================================================"

    mkdir -p "$PROJECT_DIR/images"
    cp "$SRC_DIR/"*${BAND}.TIF "$PROJECT_DIR/images/"
    echo "Copied $(ls $PROJECT_DIR/images/*.TIF | wc -l) images"

    docker run -ti --rm \
        -v "$PROJECT_DIR:/datasets/code" \
        opendronemap/odm \
        --project-path /datasets \
        --orthophoto-resolution 5 \
        --feature-quality high \
        --min-num-features 8000 \
        --mesh-size 200000 \
        --skip-3dmodel

    sudo chown -R cdac:cdac "$PROJECT_DIR"
    echo "Rain Band $BAND ($NAME) done."
done

echo "All rain season bands processed!"
