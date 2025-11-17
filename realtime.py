# asl_live_inference_top3.py
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ----------------------
# Load ASL model
# ----------------------
MODEL_PATH = "ASL.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print(model.input_shape, model.output_shape)

# Get model input size
input_shape = model.input_shape[1:3]  # e.g. (64, 64) or (128, 128)
IMG_SIZE = input_shape[0]

# Class labels (adjust to your model)
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "space", "nothing"
]

# ----------------------
# MediaPipe hands setup
# ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ----------------------
# Webcam loop
# ----------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs) * w), int(max(xs) * w)
            y_min, y_max = int(min(ys) * h), int(max(ys) * h)

            # Padding & safe crop
            pad = 20
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # Preprocess for model
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                # Predict
                preds = model.predict(hand_img, verbose=0)[0]

                # Top 3 predictions
                top3_idx = preds.argsort()[-3:][::-1]

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

                # Show top 3 stacked vertically
                for i, idx in enumerate(top3_idx):
                    if idx < len(CLASS_NAMES):
                        label = CLASS_NAMES[idx]
                    else:
                        label = str(idx)
                    confidence = preds[idx]
                    text = f"{label}: {confidence:.2f}"

                    cv2.putText(frame, text, (x_min, y_min - 10 - (i*30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("ASL Recognition - Top 3", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
