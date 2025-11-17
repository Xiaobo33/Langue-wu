#!/usr/bin/env bash
# Usage: ./make_boxes.sh input_dir output_dir lang

INPUT_DIR="$1"
OUTPUT_DIR="$2"
LANG="$3"

mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.tif; do
  [ -f "$img" ] || continue
  base=$(basename "$img" .tif)

  echo "Processing $img ..."

  # Générer le TSV
  tesseract "$img" "$OUTPUT_DIR/$base" -l "$LANG" --psm 6 tsv

  tsvfile="$OUTPUT_DIR/$base.tsv"

  # Extraire les lignes (level=4) et découper avec ImageMagick
  counter=0
  awk -F'\t' 'NR>1 && $1==4 {print $7, $8, $9, $10}' "$tsvfile" | while read left top width height; do
    ((counter++))
    out_img="$OUTPUT_DIR/${base}_line${counter}.tif"
    out_txt="$OUTPUT_DIR/${base}_line${counter}.gt.txt"

    echo "  -> cropping line $counter : $out_img"

    convert "$img" -crop "${width}x${height}+${left}+${top}" +repage "$out_img"
    echo "" > "$out_txt"
  done
done

