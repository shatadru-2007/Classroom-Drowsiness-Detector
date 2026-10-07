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
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

from drowsiness_core import DrowsinessDetector, ensure_model

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
_webcam_enabled = threading.Event()
_webcam_enabled.set()

_video_state = {
    "detected": 0,
    "drowsy": 0,
    "awake": 0,
    "percentage": 0.0,
    "alarm": False,
    "status": "empty",
    "error": None,
    "faces": [],
    "frame": 0,
    "position": 0.0,
    "duration": 0.0,
    "progress": 0.0,
}
_video_state_lock = threading.Lock()
_video_frame_lock = threading.Lock()
_video_control_lock = threading.RLock()
_video_latest_frame: bytes = b""
_video_path: Path | None = None
_video_thread: threading.Thread | None = None
_video_play_event = threading.Event()
_video_stop_event = threading.Event()
_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


# ---------------------------------------------------------------------------
# Detection thread
# ---------------------------------------------------------------------------

def detection_thread():
    global _latest_frame, _snapshot_frame
    try:
        ensure_model()
        detector = DrowsinessDetector()

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
            if not _webcam_enabled.wait(timeout=0.1):
                continue

            if _alarm_reset_event.is_set():
                alarm_active = False
                _alarm_reset_event.clear()

            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1

            # Process frame through MediaPipe + EAR + PERCLOS + SolvePnP pipeline
            res = detector.process_frame(frame)
            display = res["frame"]

            if _alarm_reset_event.is_set():
                alarm_active = False
                res["alarm"] = False
                _alarm_reset_event.clear()
            else:
                alarm_active = res["alarm"]

            _, jpeg = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            jpeg_bytes = jpeg.tobytes()

            with _frame_lock:
                _latest_frame = jpeg_bytes
                _snapshot_frame = jpeg_bytes

            stats = {
                "detected": res["detected"],
                "drowsy": res["drowsy"],
                "awake": res["awake"],
                "percentage": round(res["percentage"], 1),
                "alarm": alarm_active,
                "frame": frame_count,
                "faces": res["faces"],
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
        print(f"Detection thread error: {exc}")


def _process_uploaded_video(video_path: Path):
    global _video_latest_frame
    capture = None
    detector = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open this video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0

        ensure_model()
        detector = DrowsinessDetector()
        with _video_state_lock:
            _video_state.update({"duration": duration, "status": "running" if _video_play_event.is_set() else "paused"})

        frame_number = 0
        while not _video_stop_event.is_set():
            if not _video_play_event.wait(timeout=0.1):
                continue

            frame_started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                with _video_state_lock:
                    _video_state.update({"status": "finished", "position": duration, "progress": 100.0 if duration else 0.0})
                break

            # Use the same detector pipeline as the webcam worker.
            result = detector.process_frame(frame)
            encoded, jpeg = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not encoded:
                raise RuntimeError("Could not encode a processed video frame.")

            frame_number += 1
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if position <= 0:
                position = frame_number / fps
            progress = min(100.0, position / duration * 100.0) if duration > 0 else 0.0
            stats = {
                "detected": result["detected"],
                "drowsy": result["drowsy"],
                "awake": result["awake"],
                "percentage": round(result["percentage"], 1),
                "alarm": result["alarm"],
                "status": "running" if _video_play_event.is_set() else "paused",
                "error": None,
                "faces": result["faces"],
                "frame": frame_number,
                "position": position,
                "duration": duration,
                "progress": progress,
            }
            with _video_frame_lock:
                _video_latest_frame = jpeg.tobytes()
            with _video_state_lock:
                _video_state.update(stats)

            frame_delay = max(0.0, (1.0 / fps) - (time.monotonic() - frame_started))
            if frame_delay:
                _video_stop_event.wait(frame_delay)

    except Exception as exc:
        with _video_state_lock:
            _video_state.update({"status": "error", "error": str(exc), "alarm": False})
    finally:
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.close()
        if _video_stop_event.is_set():
            with _video_state_lock:
                _video_state.update({"status": "stopped", "alarm": False})


def _start_video_worker():
    global _video_thread
    with _video_control_lock:
        if _video_path is None:
            return False, "Upload a video first."
        if _video_thread is not None and _video_thread.is_alive():
            _video_play_event.set()
            with _video_state_lock:
                _video_state.update({"status": "running", "error": None})
            return True, None

        _video_stop_event.clear()
        _video_play_event.set()
        _video_thread = threading.Thread(
            target=_process_uploaded_video,
            args=(_video_path,),
            daemon=True,
        )
        _video_thread.start()
        with _video_state_lock:
            _video_state.update({"status": "running", "error": None})
        return True, None


def _generate_uploaded_mjpeg():
    while True:
        with _video_frame_lock:
            frame = _video_latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.033)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
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


@app.route("/video/upload", methods=["POST"])
def upload_video():
    global _video_path, _video_latest_frame
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a video file to upload."}), 400

    filename = secure_filename(uploaded.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        return jsonify({"error": "Use an MP4, AVI, MOV, MKV, or WebM video."}), 400

    if not _halt_uploaded_video():
        return jsonify({"error": "The current video is still stopping. Try again shortly."}), 409

    temp_path = None
    try:
        handle, temp_name = tempfile.mkstemp(prefix="drowsiness-video-", suffix=suffix)
        os.close(handle)
        temp_path = Path(temp_name)
        uploaded.save(temp_path)
        if temp_path.stat().st_size == 0:
            raise ValueError("The uploaded file is empty.")
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    old_path = _video_path
    _video_path = temp_path
    if old_path is not None:
        old_path.unlink(missing_ok=True)
    with _video_frame_lock:
        _video_latest_frame = b""
    with _video_state_lock:
        _video_state.update({
            "detected": 0,
            "drowsy": 0,
            "awake": 0,
            "percentage": 0.0,
            "alarm": False,
            "status": "ready",
            "error": None,
            "faces": [],
            "frame": 0,
            "position": 0.0,
            "duration": 0.0,
            "progress": 0.0,
            "filename": filename,
        })
    return jsonify({"ok": True, "filename": filename})


def _halt_uploaded_video():
    global _video_thread
    with _video_control_lock:
        worker = _video_thread
        _video_stop_event.set()
        _video_play_event.set()
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            return False
    with _video_control_lock:
        if _video_thread is worker:
            _video_thread = None
        _video_stop_event.clear()
        _video_play_event.clear()
    return True


@app.route("/video/start", methods=["POST"])
def start_video():
    started, error = _start_video_worker()
    if not started:
        return jsonify({"error": error}), 400
    return jsonify({"ok": True})


@app.route("/video/pause", methods=["POST"])
def pause_video():
    _video_play_event.clear()
    with _video_state_lock:
        if _video_state["status"] == "running":
            _video_state["status"] = "paused"
    return jsonify({"ok": True})


@app.route("/video/resume", methods=["POST"])
def resume_video():
    started, error = _start_video_worker()
    if not started:
        return jsonify({"error": error}), 400
    return jsonify({"ok": True})


@app.route("/video/stop", methods=["POST"])
def stop_video():
    if not _halt_uploaded_video():
        return jsonify({"error": "The video is still stopping. Try again shortly."}), 409
    with _video_state_lock:
        if _video_path is None:
            _video_state["status"] = "empty"
        else:
            _video_state.update({"status": "stopped", "position": 0.0, "progress": 0.0, "alarm": False})
    return jsonify({"ok": True})


@app.route("/video/restart", methods=["POST"])
def restart_video():
    global _video_latest_frame
    if not _halt_uploaded_video():
        return jsonify({"error": "The video is still stopping. Try again shortly."}), 409
    if _video_path is None:
        return jsonify({"error": "Upload a video first."}), 400
    with _video_frame_lock:
        _video_latest_frame = b""
    with _video_state_lock:
        _video_state.update({
            "detected": 0,
            "drowsy": 0,
            "awake": 0,
            "percentage": 0.0,
            "alarm": False,
            "status": "ready",
            "error": None,
            "faces": [],
            "frame": 0,
            "position": 0.0,
            "progress": 0.0,
        })
    started, error = _start_video_worker()
    if not started:
        return jsonify({"error": error}), 400
    return jsonify({"ok": True})


@app.route("/video/state")
def video_state():
    with _video_state_lock:
        return jsonify(dict(_video_state))


@app.route("/mode/<source>", methods=["POST"])
def set_mode(source):
    if source == "webcam":
        _webcam_enabled.set()
    elif source == "video":
        _webcam_enabled.clear()
    else:
        return jsonify({"error": "Unknown video source."}), 400
    return jsonify({"ok": True})


@app.route("/video/feed")
def uploaded_video_feed():
    return Response(
        _generate_uploaded_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video/stats")
def uploaded_video_stats():
    def generate():
        while True:
            with _video_state_lock:
                stats = dict(_video_state)
            yield f"data: {json.dumps(stats)}\n\n"
            time.sleep(0.2)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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