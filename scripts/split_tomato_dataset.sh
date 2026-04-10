#!/bin/bash
set -euo pipefail

SOURCE_DIR="data/raw/plant_disease_labeled_images"
TARGET_TRAIN="data/processed/train"
TARGET_VAL="data/processed/val"
TRAIN_RATIO=0.8

mkdir -p "$TARGET_TRAIN"
mkdir -p "$TARGET_VAL"

echo "Preparing Tomato dataset..."
echo "Source: $SOURCE_DIR"
echo "Train:  $TARGET_TRAIN"
echo "Val:    $TARGET_VAL"
echo

# Find all tomato-related class folders (case-insensitive)
mapfile -d '' class_dirs < <(find "$SOURCE_DIR" -type d -iname "*tomato*" -print0)

for class_dir in "${class_dirs[@]}"; do
    class_name=$(basename "$class_dir")
    echo "Processing class: $class_name"

    mkdir -p "$TARGET_TRAIN/$class_name"
    mkdir -p "$TARGET_VAL/$class_name"

    # Find all images robustly (handles spaces, unicode, weird chars)
    mapfile -d '' images < <(
        find "$class_dir" -maxdepth 1 -type f \
            \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
            -print0
    )

    total=${#images[@]}
    if [[ $total -eq 0 ]]; then
        echo "  ⚠️  No images found — skipping"
        continue
    fi

    # Compute train count
    train_count=$(printf "%.0f" "$(echo "$total * $TRAIN_RATIO" | bc)")

    # Shuffle images safely
    mapfile -t shuffled < <(printf "%s\n" "${images[@]}" | shuf)

    # Split arrays
    train_imgs=("${shuffled[@]:0:train_count}")
    val_imgs=("${shuffled[@]:train_count}")

    # Copy train images
    for img in "${train_imgs[@]}"; do
        cp "$img" "$TARGET_TRAIN/$class_name/"
    done

    # Copy val images
    for img in "${val_imgs[@]}"; do
        cp "$img" "$TARGET_VAL/$class_name/"
    done

    echo "  → $train_count train images"
    echo "  → $((${#val_imgs[@]})) val images"
    echo
done

echo "Dataset preparation complete."
