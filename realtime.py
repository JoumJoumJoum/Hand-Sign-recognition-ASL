import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import serial
import time

# ----------------------
# Arduino Serial Setup
# ----------------------
# ⚠️ CHANGE THIS to whatever shows as your Arduino port in Tools > Port
ARDUINO_PORT = "COM4"   # e.g. "COM4", "COM5", etc.
BAUD_RATE = 9600

arduino = None
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give Arduino time to reset
    print(f"✅ Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    print(f"⚠️ Error connecting to Arduino: {e}")
    arduino = None


def send_to_arduino(text: str):
    """Send a short string to Arduino, if serial is available."""
    if arduino is None:
        return
    try:
        # LCD is 16 chars wide → trim it
        text = str(text)[:16]
        arduino.write((text + "\n").encode("utf-8"))
        print(f"➡️ Sent to Arduino: {repr(text)}")
    except Exception as e:
        print(f"⚠️ Error sending to Arduino: {e}")


# ----------------------
# Load ASL model
# ----------------------
MODEL_PATH = "ASL.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model shapes:", model.input_shape, model.output_shape)

# Get model input size
input_shape = model.input_shape[1:3]
IMG_SIZE = input_shape[0]

# Class labels (index -> label)
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
cap = cv2.VideoCapture(0)  # if webcam issues, try 0 or 1 or add cv2.CAP_DSHOW on Windows

last_letter = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(xs) * w), int(max(xs) * w)
            y_min, y_max = int(min(ys) * h), int(max(ys) * h)

            pad = 20
            x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
            x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                preds = model.predict(hand_img, verbose=0)[0]
                top3_idx = preds.argsort()[-3:][::-1]

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                top_idx = top3_idx[0]
                top_letter = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
                top_confidence = preds[top_idx]

                # ----------------------
                # Send only meaningful labels to Arduino LCD
                # ----------------------
                send_char = None
                if top_letter in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                    send_char = top_letter
                elif top_letter == "space":
                    send_char = " "       # actual space
                # ignore "del" and "nothing" for now on LCD

                # Only send when:
                # 1) Confidence is high, and
                # 2) Letter changed
                if send_char is not None and top_confidence > 0.7 and top_letter != last_letter:
                    send_to_arduino(send_char)
                    last_letter = top_letter

                # Show top 3 predictions on the OpenCV window
                for i, idx in enumerate(top3_idx):
                    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                    confidence = preds[idx]
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x_min, y_min - 10 - (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

    cv2.imshow("ASL Recognition - Top 3", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
if arduino is not None and arduino.is_open:
    arduino.close()
    print("🔌 Closed Arduino serial")
