"""
Script 3: Evaluate SVHN Model
- Loads the saved model and test data
- Computes per-digit accuracy AND full-sequence accuracy
  (full-sequence = the entire number must be correct)
- Prints a summary table and saves results to results.txt

Usage:
    python 3_evaluate.py

Requirements:
    svhn_model.keras       (from 2_train.py)
    test_images.npy        (from 1_preprocess.py)
    test_labels.npy        (from 1_preprocess.py)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "svhn_model.keras"
MAX_DIGITS = 5
BATCH_SIZE = 128
# ────────────────────────────────────────────────────────────────────────────

print("Loading model ...")
model = keras.models.load_model(MODEL_PATH)

print("Loading test data ...")
X_test = np.load("test_images.npy").astype("float32") / 255.0
y_test = np.load("test_labels.npy")   # shape (N, MAX_DIGITS+1)

print(f"Test set size: {len(X_test)}")

# ── Run predictions ───────────────────────────────────────────────────────────
outputs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
# outputs is a list: [length_probs, d1_probs, d2_probs, d3_probs, d4_probs, d5_probs]

pred_length = np.argmax(outputs[0], axis=1)
pred_digits = np.stack([np.argmax(outputs[i+1], axis=1) for i in range(MAX_DIGITS)], axis=1)
# pred_digits shape: (N, MAX_DIGITS)

true_length = y_test[:, 0]
true_digits = y_test[:, 1:]

# ── Metrics ───────────────────────────────────────────────────────────────────

# 1) Length accuracy
length_acc = np.mean(pred_length == true_length)

# 2) Per-position digit accuracy (only for positions within the true length)
digit_accs = []
for pos in range(MAX_DIGITS):
    mask = true_length > pos   # only samples where this position is real
    if mask.sum() == 0:
        digit_accs.append(float("nan"))
    else:
        acc = np.mean(pred_digits[mask, pos] == true_digits[mask, pos])
        digit_accs.append(acc)

# 3) Full-sequence accuracy
#    The entire predicted number must match ground truth.
#    For each sample: length must match AND all digits up to that length must match.
correct_seq = np.ones(len(X_test), dtype=bool)
correct_seq &= (pred_length == true_length)
for pos in range(MAX_DIGITS):
    mask = true_length > pos
    correct_seq[mask] &= (pred_digits[mask, pos] == true_digits[mask, pos])
seq_acc = np.mean(correct_seq)

# ── Print results ─────────────────────────────────────────────────────────────
lines = []
lines.append("=" * 50)
lines.append("       SVHN Evaluation Results")
lines.append("=" * 50)
lines.append(f"Test samples         : {len(X_test)}")
lines.append(f"Length accuracy      : {length_acc:.4f}  ({length_acc*100:.2f}%)")
lines.append("")
lines.append("Per-digit accuracy (active positions only):")
for i, acc in enumerate(digit_accs):
    if np.isnan(acc):
        lines.append(f"  Digit position {i+1}  : N/A (no samples)")
    else:
        lines.append(f"  Digit position {i+1}  : {acc:.4f}  ({acc*100:.2f}%)")
lines.append("")
lines.append(f"Full-sequence accuracy: {seq_acc:.4f}  ({seq_acc*100:.2f}%)")
lines.append("=" * 50)

report = "\n".join(lines)
print(report)

with open("results.txt", "w") as f:
    f.write(report + "\n")
print("\nResults saved to results.txt")


# ── Sample predictions (first 10 test images) ─────────────────────────────────
def decode_number(length, digits):
    """Convert model output to a human-readable number string."""
    n = int(length)
    if n == 0:
        return "(none)"
    parts = []
    for i in range(n):
        d = int(digits[i])
        parts.append("0" if d == 10 else str(d))
    return "".join(parts)

print("\nSample predictions (first 10):")
print(f"{'Index':<8} {'True':>10} {'Predicted':>12} {'Correct':>8}")
print("-" * 42)
for i in range(min(10, len(X_test))):
    true_str = decode_number(true_length[i], true_digits[i])
    pred_str = decode_number(pred_length[i], pred_digits[i])
    ok = "✓" if correct_seq[i] else "✗"
    print(f"{i:<8} {true_str:>10} {pred_str:>12} {ok:>8}")
