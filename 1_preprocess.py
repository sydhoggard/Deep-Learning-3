"""
Script 1: Preprocess SVHN Data
- Reads digitStruct.json
- Crops images to the bounding box region (with padding)
- Resizes to 64x64
- Saves labels as numpy arrays
- Run FIRST before training

Usage:
    python 1_preprocess.py

Output:
    train_images.npy, train_labels.npy
    test_images.npy,  test_labels.npy
"""

import os
import json
import numpy as np
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_DIR = "train"          # folder with training PNGs + digitStruct.json
TEST_DIR  = "test"           # folder with test PNGs  + digitStruct.json
IMG_SIZE  = (96, 96)         # resize target
MAX_DIGITS = 5               # SVHN has at most 5 digits per image
PADDING   = 0.15             # fractional padding added around the bounding box crop
# ────────────────────────────────────────────────────────────────────────────


def load_digit_struct(folder):
    """Load digitStruct.json and return list of dicts."""
    path = os.path.join(folder, "digitStruct.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data


def crop_and_resize(img_path, boxes, padding=PADDING, size=IMG_SIZE):
    """
    Crop the image to cover all digit bounding boxes (+ padding),
    then resize to `size`.
    Returns a numpy uint8 array of shape (H, W, 3).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Compute union bounding box across all digits
    tops    = [b["top"]  for b in boxes]
    lefts   = [b["left"] for b in boxes]
    bottoms = [b["top"]  + b["height"] for b in boxes]
    rights  = [b["left"] + b["width"]  for b in boxes]

    y1 = min(tops);    y2 = max(bottoms)
    x1 = min(lefts);   x2 = max(rights)

    # Add padding
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0,   x1 - pad_x)
    y1 = max(0,   y1 - pad_y)
    x2 = min(w,   x2 + pad_x)
    y2 = min(h,   y2 + pad_y)

    img = img.crop((x1, y1, x2, y2)).resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def build_label_vector(boxes, max_digits=MAX_DIGITS):
    """
    Returns an int array of length max_digits+1:
      [num_digits, d1, d2, d3, d4, d5]
    Digits past the actual count are filled with 0 (null class).
    Label 10 in the JSON means digit '0' — kept as-is for the model.
    """
    n = min(len(boxes), max_digits)
    # Sort boxes left-to-right so digits are in reading order
    sorted_boxes = sorted(boxes, key=lambda b: b["left"])
    label = np.zeros(max_digits + 1, dtype=np.int32)
    label[0] = n
    for i, box in enumerate(sorted_boxes[:n]):
        label[i + 1] = int(box["label"])
    return label


def process_folder(folder, split_name):
    records = load_digit_struct(folder)
    images  = []
    labels  = []
    skipped = 0

    print(f"Processing {len(records)} images in '{folder}' ...")
    for i, rec in enumerate(records):
        img_path = os.path.join(folder, rec["filename"])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        boxes = rec["boxes"]
        # Skip images with more digits than we handle
        if len(boxes) > MAX_DIGITS:
            skipped += 1
            continue

        arr = crop_and_resize(img_path, boxes)
        lbl = build_label_vector(boxes)
        images.append(arr)
        labels.append(lbl)

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(records)} done, skipped so far: {skipped}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    np.save(f"{split_name}_images.npy", images)
    np.save(f"{split_name}_labels.npy", labels)
    print(f"Saved {split_name}_images.npy  shape={images.shape}")
    print(f"Saved {split_name}_labels.npy  shape={labels.shape}")
    print(f"Skipped {skipped} images.\n")


if __name__ == "__main__":
    process_folder(TRAIN_DIR, "train")
    process_folder(TEST_DIR,  "test")
    print("Preprocessing complete.")
