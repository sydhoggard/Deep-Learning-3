import numpy as np
from tensorflow import keras

MODEL_PATH = "svhn_cnn_model_improved.keras"
MAX_DIGITS = 5
BATCH_SIZE = 128

def decode_number(length, digits):
    n = int(length)

    if n == 0:
        return "(none)"

    result = []

    for i in range(n):
        d = int(digits[i])
        if d == 10:
            result.append("0")
        elif d == 0:
            result.append("_")
        else:
            result.append(str(d))

    return "".join(result)

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

print("Loading test data...")
X_test = np.load("test_images.npy").astype("float32") / 255.0
y_test = np.load("test_labels.npy")

outputs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

pred_length = np.argmax(outputs[0], axis=1)
pred_digits = np.stack(
    [np.argmax(outputs[i + 1], axis=1) for i in range(MAX_DIGITS)],
    axis=1
)

true_length = y_test[:, 0]
true_digits = y_test[:, 1:]

length_acc = np.mean(pred_length == true_length)

digit_accs = []

for pos in range(MAX_DIGITS):
    mask = true_length > pos

    if mask.sum() == 0:
        digit_accs.append(float("nan"))
    else:
        digit_accs.append(np.mean(pred_digits[mask, pos] == true_digits[mask, pos]))

correct_seq = pred_length == true_length

for pos in range(MAX_DIGITS):
    mask = true_length > pos
    correct_seq[mask] &= pred_digits[mask, pos] == true_digits[mask, pos]

seq_acc = np.mean(correct_seq)

lines = []
lines.append("=" * 55)
lines.append("SVHN Evaluation Results")
lines.append("=" * 55)
lines.append(f"Test samples           : {len(X_test)}")
lines.append(f"Length accuracy        : {length_acc:.4f} ({length_acc * 100:.2f}%)")
lines.append("")

for i, acc in enumerate(digit_accs):
    lines.append(f"Digit position {i + 1}      : {acc:.4f} ({acc * 100:.2f}%)")

lines.append("")
lines.append(f"Full-sequence accuracy : {seq_acc:.4f} ({seq_acc * 100:.2f}%)")
lines.append("=" * 55)

report = "\n".join(lines)
print(report)

with open("results.txt", "w") as f:
    f.write(report + "\n")

print("\nSample predictions:")
print(f"{'Index':<8} {'True':>12} {'Predicted':>12} {'Correct':>8}")
print("-" * 45)

for i in range(20):
    true_str = decode_number(true_length[i], true_digits[i])
    pred_str = decode_number(pred_length[i], pred_digits[i])
    ok = "yes" if correct_seq[i] else "no"
    print(f"{i:<8} {true_str:>12} {pred_str:>12} {ok:>8}")
