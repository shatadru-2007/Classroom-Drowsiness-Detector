"""
app.py – Flask web server for the Classroom Drowsiness Detector UI.

Routes:
  GET /           → serve the dashboard HTML
  GET /video_feed → MJPEG annotated webcam stream
  GET /stats      → Server-Sent Events with live JSON stats
  POST /reset     → reset alarm state
  GET /snapshot   → download a JPEG snapshot
"""

import io
import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, send_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://huggingface.co/notgoodkeeper/"
    "cnn-based-drowsiness-detection/resolve/main/model.onnx"
)
MODEL_PATH = Path("models/drowsiness_model.onnx")

INPUT_SIZE = 412
DROWSY_THRESHOLD = 0.50
ALARM_THRESHOLD = 0.50
MIN_FACE_SIZE = (55, 55)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_state = {
    "detected": 0,
    "drowsy": 0,
    "awake": 0,
    "percentage": 0.0,
    "alarm": False,
    "running": False,
    "error": None,
    "faces": [],
    "frame_count": 0,
}
_state_lock = threading.Lock()

_latest_frame: bytes = b""
_snapshot_frame: bytes = b""
_frame_lock = threading.Lock()

_stats_queue: queue.Queue = queue.Queue(maxsize=60)
_alarm_reset_event = threading.Event()

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    print("Downloading ONNX model from Hugging Face…")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    if MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Downloaded model file appears incomplete.")


def make_session():
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def preprocess(face_bgr):
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

    drowsy_score = predicted_class = confidence = None

    if len(outputs) >= 1:
        predicted_class = scalar_from_output(outputs[0])
    if len(outputs) >= 2:
        confidence = scalar_from_output(outputs[1])
    if len(outputs) >= 3:
        reg = np.asarray(outputs[2]).reshape(-1)
        if reg.size >= 1:
            drowsy_score = float(reg[0])

    if drowsy_score is not None and np.isfinite(drowsy_score):
        drowsy = drowsy_score >= DROWSY_THRESHOLD
        score = drowsy_score
    else:
        pc = float(np.asarray(predicted_class).reshape(-1)[0])
        conf = (float(np.asarray(confidence).reshape(-1)[0])
                if confidence is not None else 1.0)
        drowsy = pc > 0.5
        score = conf if drowsy else 1.0 - conf

    return drowsy, float(score)


# ---------------------------------------------------------------------------
# Detection thread
# ---------------------------------------------------------------------------

def _draw_overlay(display, detected, drowsy_count, percentage, alarm_on):
    h, w = display.shape[:2]
    # Bottom status bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, h - 56), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    status_text = (
        f"Detected: {detected}   Drowsy: {drowsy_count}   "
        f"Awake: {detected - drowsy_count}   Ratio: {percentage:.0%}"
    )
    cv2.putText(display, status_text, (16, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 255), 2)
    if alarm_on:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
        banner_overlay = display.copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, 52), (0, 0, 180), -1)
        cv2.addWeighted(banner_overlay, 0.65, display, 0.35, 0, display)
        cv2.putText(display, "! CLASSROOM DROWSINESS ALERT > 50% !",
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 220, 50), 2)


def detection_thread():
    global _latest_frame, _snapshot_frame
    try:
        ensure_model()
        session = make_session()
        face_detector = cv2.CascadeClassifier(CASCADE_PATH)
        if face_detector.empty():
            raise RuntimeError("Haar face detector could not be loaded.")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with _state_lock:
            _state["running"] = True

        alarm_active = False
        frame_count = 0

        while True:
            if _alarm_reset_event.is_set():
                alarm_active = False
                _alarm_reset_event.clear()

            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            display = frame.copy()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray_frame, scaleFactor=1.08, minNeighbors=5, minSize=MIN_FACE_SIZE
            )

            drowsy_count = 0
            detected_count = len(faces)
            face_results = []

            for idx, (x, y, w, h) in enumerate(faces, start=1):
                pad_x = int(w * 0.15)
                pad_top = int(h * 0.20)
                pad_bottom = int(h * 0.10)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_top)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_bottom)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    drowsy, score = classify_face(session, face_crop)
                except Exception:
                    continue

                if drowsy:
                    drowsy_count += 1
                    box_color = (50, 50, 255)
                    status_label = "DROWSY"
                else:
                    box_color = (50, 210, 50)
                    status_label = "AWAKE"

                face_results.append({
                    "id": idx,
                    "label": f"Student {idx}",
                    "status": status_label,
                    "score": round(score * 100, 1),
                })

                cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
                label = f"S{idx}: {status_label} {score:.0%}"
                cv2.putText(display, label, (x, max(25, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            percentage = drowsy_count / detected_count if detected_count else 0.0
            alarm_on = detected_count > 0 and percentage > ALARM_THRESHOLD

            if alarm_on and not alarm_active:
                alarm_active = True
            elif not alarm_on:
                alarm_active = False

            _draw_overlay(display, detected_count, drowsy_count, percentage, alarm_on)

            _, jpeg = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82]
            )
            jpeg_bytes = jpeg.tobytes()

            with _frame_lock:
                _latest_frame = jpeg_bytes
                _snapshot_frame = jpeg_bytes

            stats = {
                "detected": detected_count,
                "drowsy": drowsy_count,
                "awake": detected_count - drowsy_count,
                "percentage": round(percentage * 100, 1),
                "alarm": alarm_on,
                "frame": frame_count,
                "faces": face_results,
                "ts": time.time(),
            }
            with _state_lock:
                _state.update(stats)
                _state["running"] = True

            try:
                _stats_queue.put_nowait(stats)
            except queue.Full:
                pass

        cap.release()

    except Exception as exc:
        with _state_lock:
            _state["error"] = str(exc)
            _state["running"] = False
        print(f"Detection error: {exc}")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
_UI_HTML = (Path(__file__).with_name("ui.html")).read_text(encoding="utf-8")


@app.route("/")
def index():
    return Response(_UI_HTML, mimetype="text/html")


def _generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _generate_sse():
    while True:
        try:
            stats = _stats_queue.get(timeout=2.0)
            yield f"data: {json.dumps(stats)}\n\n"
        except queue.Empty:
            with _state_lock:
                current = dict(_state)
            yield f"data: {json.dumps(current)}\n\n"


@app.route("/stats")
def stats_stream():
    return Response(
        _generate_sse(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reset", methods=["POST"])
def reset_alarm():
    _alarm_reset_event.set()
    return jsonify({"ok": True})


@app.route("/snapshot")
def snapshot():
    with _frame_lock:
        frame = _snapshot_frame
    if not frame:
        return jsonify({"error": "No frame available yet"}), 404
    buf = io.BytesIO(frame)
    buf.seek(0)
    fname = f"snapshot_{int(time.time())}.jpg"
    return send_file(buf, mimetype="image/jpeg",
                     as_attachment=True, download_name=fname)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dt = threading.Thread(target=detection_thread, daemon=True)
    dt.start()

    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
