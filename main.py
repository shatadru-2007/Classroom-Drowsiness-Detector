import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_URL = (
    "https://huggingface.co/notgoodkeeper/"
    "cnn-based-drowsiness-detection/resolve/main/model.onnx"
)
MODEL_PATH = Path("models/drowsiness_model.onnx")

INPUT_SIZE = 412
DROWSY_THRESHOLD = 0.50
ALARM_THRESHOLD = 0.50
MIN_FACE_SIZE = (55, 55)

# OpenCV ships this pretrained face detector with the Python package.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    print("Pretrained ONNX model not found.")
    print("Downloading from Hugging Face...")
    print(MODEL_URL)
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pretrained model. "
            "Check your internet connection and try again."
        ) from exc

    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model file appears incomplete.")


def make_session():
    # CPUExecutionProvider avoids CUDA/GPU requirements.
    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )


def preprocess(face_bgr):
    # The model card specifies a 412x412 RGB input. The training pipeline
    # used grayscale/CLAHE preprocessing and replicated the grayscale image
    # to 3 channels.
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor


def scalar_from_output(value):
    arr = np.asarray(value).squeeze()
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def classify_face(session, face_bgr):
    input_name = session.get_inputs()[0].name
    tensor = preprocess(face_bgr)
    outputs = session.run(None, {input_name: tensor})

    # Model card output order:
    # predicted_class, confidence, regressions
    # regressions[0] = normalized drowsiness score.
    drowsy_score = None
    predicted_class = None
    confidence = None

    if len(outputs) >= 1:
        predicted_class = scalar_from_output(outputs[0])
    if len(outputs) >= 2:
        confidence = scalar_from_output(outputs[1])
    if len(outputs) >= 3:
        reg = np.asarray(outputs[2]).reshape(-1)
        if reg.size >= 1:
            drowsy_score = float(reg[0])

    # Prefer the model's explicit drowsiness score when available.
    if drowsy_score is not None and np.isfinite(drowsy_score):
        drowsy = drowsy_score >= DROWSY_THRESHOLD
        score = drowsy_score
    else:
        # Fallback if a provider/version exposes only class + confidence.
        pc = float(np.asarray(predicted_class).reshape(-1)[0])
        conf = float(np.asarray(confidence).reshape(-1)[0]) if confidence is not None else 1.0
        drowsy = pc > 0.5
        score = conf if drowsy else 1.0 - conf

    return drowsy, float(score)


def play_alarm():
    # Built-in Windows sound; no extra audio package required.
    try:
        import winsound
        winsound.Beep(1200, 180)
        winsound.Beep(900, 180)
    except Exception:
        pass


def main():
    ensure_model()

    print("Loading ONNX Runtime model...")
    session = make_session()

    print("Model inputs:")
    for item in session.get_inputs():
        print(" ", item.name, item.shape, item.type)
    print("Model outputs:")
    for item in session.get_outputs():
        print(" ", item.name, item.shape, item.type)

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        raise RuntimeError("OpenCV Haar face detector could not be loaded.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    alarm_active = False
    last_alarm = 0.0
    frame_count = 0

    print("\nRunning. Press Q to quit, S to save a screenshot, R to reset alarm.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a webcam frame.")
                break

            frame_count += 1
            display = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
            )

            drowsy_count = 0
            detected_count = len(faces)

            for idx, (x, y, w, h) in enumerate(faces, start=1):
                # Add a little context around the face.
                pad_x = int(w * 0.15)
                pad_top = int(h * 0.20)
                pad_bottom = int(h * 0.10)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_top)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_bottom)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                try:
                    drowsy, score = classify_face(session, face)
                except Exception as exc:
                    cv2.putText(
                        display, "MODEL ERROR", (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )
                    print(f"Model error on face {idx}: {exc}")
                    continue

                if drowsy:
                    drowsy_count += 1
                    label = f"Student {idx}: DROWSY {score:.0%}"
                    box_color = (0, 0, 255)
                else:
                    label = f"Student {idx}: AWAKE {score:.0%}"
                    box_color = (0, 180, 0)

                cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(
                    display, label, (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2
                )

            percentage = (
                drowsy_count / detected_count if detected_count else 0.0
            )
            alarm_should_be_on = detected_count > 0 and percentage > ALARM_THRESHOLD

            if alarm_should_be_on:
                cv2.rectangle(
                    display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1),
                    (0, 0, 255), 8
                )
                cv2.putText(
                    display, "ALARM: CLASSROOM DROWSINESS > 50%",
                    (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
                )
                now = time.time()
                if (not alarm_active) or (now - last_alarm > 2.0):
                    play_alarm()
                    last_alarm = now
                alarm_active = True
            else:
                alarm_active = False

            status = (
                f"Detected: {detected_count} | Drowsy: {drowsy_count} | "
                f"Drowsy %: {percentage:.0%}"
            )
            cv2.putText(
                display, status, (20, display.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )

            rule = "Alarm rule: > 50% drowsy"
            cv2.putText(
                display, rule, (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            cv2.imshow("Classroom Drowsiness Detection - ONNX", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"classroom_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, display)
                print(f"Saved {filename}")
            elif key == ord("r"):
                alarm_active = False
                last_alarm = 0.0
                print("Alarm state reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
