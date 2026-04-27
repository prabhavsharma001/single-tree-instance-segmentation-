#!/bin/bash

SRC_DIR="/home/cdac/Downloads/RGB and Multispectral images dataset ( Álamos, Sonora)/Dry Season/Dry Season Multi"
OUT_DIR="/home/cdac/Desktop/detectree2/odm_projects/dry_multi/images"

echo "Copying EXIF GPS tags from band-1 originals to stacked TIFs..."
count=0

for stacked in "$OUT_DIR"/*.TIF; do
    img_id=$(basename "$stacked" | sed 's/DJI_\(...\)_multi\.TIF/\1/')
    src_band1="$SRC_DIR/DJI_${img_id}1.TIF"

    if [ -f "$src_band1" ]; then
        exiftool -overwrite_original \
            -TagsFromFile "$src_band1" \
            -GPS:all \
            -exif:all \
            "$stacked" > /dev/null 2>&1
        count=$((count + 1))
        if [ $((count % 10)) -eq 0 ]; then
            echo "  Processed $count/98..."
        fi
    fi
done

echo "Done. EXIF copied to $count files."
