import os
import json
import numpy as np
from PIL import Image

TRAIN_DIR = "train"
TEST_DIR = "test"
IMG_SIZE = (96, 96)
MAX_DIGITS = 5
PADDING = 0.25

def load_digit_struct(folder):
    with open(os.path.join(folder, "digitStruct.json"), "r") as f:
        return json.load(f)

def crop_and_resize(img_path, boxes):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    tops = [float(b["top"]) for b in boxes]
    lefts = [float(b["left"]) for b in boxes]
    bottoms = [float(b["top"]) + float(b["height"]) for b in boxes]
    rights = [float(b["left"]) + float(b["width"]) for b in boxes]

    x1, y1 = min(lefts), min(tops)
    x2, y2 = max(rights), max(bottoms)

    pad_x = (x2 - x1) * PADDING
    pad_y = (y2 - y1) * PADDING

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    img = img.crop((x1, y1, x2, y2))
    img = img.resize(IMG_SIZE, Image.BILINEAR)

    return np.array(img, dtype=np.uint8)

def build_label_vector(boxes):
    boxes = sorted(boxes, key=lambda b: float(b["left"]))

    label = np.zeros(MAX_DIGITS + 1, dtype=np.int32)
    label[0] = min(len(boxes), MAX_DIGITS)

    for i, box in enumerate(boxes[:MAX_DIGITS]):
        label[i + 1] = int(box["label"])

    return label

def process_folder(folder, split_name):
    records = load_digit_struct(folder)

    images = []
    labels = []
    skipped = 0

    print(f"Processing {len(records)} images in {folder}...")

    for i, rec in enumerate(records):
        img_path = os.path.join(folder, rec["filename"])
        boxes = rec["boxes"]

        if not os.path.exists(img_path) or len(boxes) == 0 or len(boxes) > MAX_DIGITS:
            skipped += 1
            continue

        try:
            images.append(crop_and_resize(img_path, boxes))
            labels.append(build_label_vector(boxes))
        except Exception as e:
            skipped += 1
            print("Skipped:", rec["filename"], e)

        if (i + 1) % 2000 == 0:
            print(f"{i+1}/{len(records)} done, skipped={skipped}")

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    np.save(f"{split_name}_images.npy", images)
    np.save(f"{split_name}_labels.npy", labels)

    print(f"{split_name}_images.npy:", images.shape)
    print(f"{split_name}_labels.npy:", labels.shape)
    print("Skipped:", skipped)

process_folder("train", "train")
process_folder("test", "test")
