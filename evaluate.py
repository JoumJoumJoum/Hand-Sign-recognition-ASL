import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt

# ----------------------------
# 1. CONFIG
# ----------------------------
MODEL_PATH = "hand_sign_model.h5"  # your saved model
DATA_DIR   = "data"                # data/A, data/B, ...

BATCH_SIZE = 32
VAL_SPLIT  = 0.2                   # 20% of data used as "test"

# ----------------------------
# 2. LOAD MODEL
# ----------------------------
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Auto-detect input size from the model
input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
if len(input_shape) == 4:
    _, h, w, c = input_shape
    IMG_SIZE = (h, w)
else:
    # Fallback
    IMG_SIZE = (224, 224)

print(f"[INFO] Model input shape: {input_shape}, using IMG_SIZE={IMG_SIZE}")

# ----------------------------
# 3. DATA GENERATOR (AUTO SPLIT)
# ----------------------------
print("[INFO] Preparing data generator (using validation_split)...")

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=VAL_SPLIT
)

test_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

y_true = test_generator.classes

class_indices = test_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print("\n[INFO] Classes found:")
print(class_indices)

# ----------------------------
# 4. PREDICT
# ----------------------------
print("\n[INFO] Predicting on validation/test subset...")
y_prob = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# ----------------------------
# 5. METRICS
# ----------------------------
acc = accuracy_score(y_true, y_pred)
print("\n=== Overall Accuracy ===")
print(f"Accuracy: {acc * 100:.2f}%")

try:
    top3_acc = top_k_accuracy_score(y_true, y_prob, k=3)
    print(f"Top-3 Accuracy: {top3_acc * 100:.2f}%")
except Exception as e:
    print("Top-3 accuracy not computed:", e)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("\n=== Confusion Matrix (raw counts) ===")
print(cm)

# ----------------------------
# 6. PLOT + SAVE CONFUSION MATRIX
# ----------------------------
def plot_and_save_confusion_matrix(cm, class_names, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[INFO] Confusion matrix image saved as: {filename}")
    plt.show()

print("\n[INFO] Plotting and saving confusion matrix...")
plot_and_save_confusion_matrix(cm, class_names)

print("\n[INFO] Evaluation complete.")
