"""
drowsiness_core.py – High-Accuracy Classroom Drowsiness Detection Pipeline using 
MediaPipe Face Landmarker, OpenCV, and NumPy.

Pipeline features:
  1. Multi-face landmark detection via MediaPipe Face Landmarker with dual-pass 
     contrast/illumination enhancement (CLAHE) for distant/shadowed classroom faces.
  2. Eye Aspect Ratio (EAR) calculation for left and right eyes.
  3. PERCLOS (Percentage of Eye Closure over time) sliding window calculation.
  4. 3D Head Pose estimation (Pitch, Yaw, Roll) via OpenCV solvePnP.
  5. Multimodal drowsiness scoring (low EAR + downward pitch/slouch + duration).
  6. Temporal hysteresis & state smoothing to avoid rapid flickering.
  7. Independent multi-student tracking with OpenCV Haar Cascade crop-recovery fallback.
  8. Fallback Severe-Slouch Mode: Multi-feature CV silhouette verification 
     (edge density, Sobel gradient, HSV texture, spatial structure) when face is lost.
"""

import math
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("models/face_landmarker.task")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EAR_CLOSED_THRESHOLD = 0.21
EAR_OPEN_THRESHOLD = 0.24
PITCH_DOWN_THRESHOLD = -20.0  # degrees (relative negative = head tilted down / slouching)
PITCH_UP_THRESHOLD = -8.0

PERCLOS_WINDOW_SIZE = 150      # ~5 seconds at 30 fps
PERCLOS_DROWSY_RATIO = 0.40   # 40% closed frames in window = drowsy
PERCLOS_AWAKE_RATIO = 0.60

CONSECUTIVE_DROWSY_FRAMES = 150 # 5 seconds
CONSECUTIVE_AWAKE_FRAMES = 1

MAX_TRACKER_MISSING_FRAMES = 45
MAX_SLOUCH_MISSING_FRAMES = 120
MAX_MATCH_DISTANCE_PX = 150.0

SLOUCH_PERSISTENCE_FRAMES = 150  # ~5 seconds of sustained slouching required
SLOUCH_DOWNWARD_SHIFT_PX = 20.0  # minimum downward pixel displacement threshold

# --- Small-face (classroom camera) detection -------------------------------
# MediaPipe's built-in face detector downsizes the whole frame to a tiny input,
# so faces narrower than ~10% of the frame width are never found. We therefore
# (a) scan overlapping, upscaled tiles to DISCOVER faces every few frames and
# (b) re-run the landmarker on an upscaled crop around each KNOWN face every frame.
DISCOVERY_INTERVAL_FRAMES = 30      # how often to start a (background) tiled scan
ROI_REFRESH_FRAMES = 3              # re-run landmarks on each tracked face every N frames
TILE_FRACTIONS = (0.20, 0.35)       # tile width as a fraction of frame width
TILE_OVERLAP = 0.5
TILE_INPUT_SIZE = 720               # tiles are resized to this before detection
ROI_MARGIN = 0.8                    # crop margin around a tracked face (x face size)
ROI_INPUT_SIZE = 320                # tracked-face crops are resized to this
MIN_FACE_PX = 10                    # reject detections narrower than this

# On-face drowsiness score (%) thresholds - these decide DROWSY / AWAKE
SCORE_DROWSY_THRESHOLD = 70.0   # AWAKE -> DROWSY when score >= this
SCORE_AWAKE_THRESHOLD = 60.0    # DROWSY -> AWAKE when score < this (60-70 = hold zone)
MIN_FRAMES_FOR_DECISION = 30    # warm-up: ignore the score for a new student's first ~1 s

# Closed / lowered eyes with an UPRIGHT head usually means "looking down at a book",
# not sleeping. When enabled, eye closure only counts once the head has also sagged
# by HEAD_DROP_DEG below that student's own upright posture.
REQUIRE_HEAD_DROP_FOR_EYES = True
HEAD_DROP_DEG = 12.0            # degrees below own upright pitch = "head has dropped"
HEAD_BASELINE_FRAMES = 30       # frames used to learn each student's upright pitch
HEAD_SMOOTH_FRAMES = 6          # frames averaged when judging head drop (lower = reacts faster)

# Fast confirmation: eyes closed AND head dropped for this many frames in a row
# flags the student immediately instead of waiting for the 5 s average to climb.
FAST_DROWSY_FRAMES = 8          # ~0.3 s at 30 fps

ALARM_THRESHOLD_PERCENT = 0.40  # > 40% students drowsy triggers alarm
ALARM_PERSISTENCE_FRAMES = 60   # ~2 seconds sustained before triggering global alarm

# MediaPipe 468 landmark indices
# Left eye: [corner_left, top1, top2, corner_right, bot2, bot1]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Right eye: [corner_left, top1, top2, corner_right, bot2, bot1]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Head pose 3D model points in camera space (+X right, +Y down, +Z away)
HEAD_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (1)
    (0.0, 110.0, -50.0),         # Chin (152)
    (-75.0, -75.0, -75.0),       # Left eye outer corner (33)
    (75.0, -75.0, -75.0),        # Right eye outer corner (263)
    (-40.0, 50.0, -50.0),        # Left mouth corner (61)
    (40.0, 50.0, -50.0)          # Right mouth corner (291)
], dtype=np.float64)

HEAD_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]


def ensure_model():
    """Download MediaPipe FaceLandmarker task file if missing or incomplete."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 500_000:
        return
    print("Downloading MediaPipe Face Landmarker model asset...")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    if MODEL_PATH.stat().st_size < 500_000:
        raise RuntimeError("Downloaded face landmarker model asset is incomplete.")
    print("Face Landmarker model ready.")


# ---------------------------------------------------------------------------
# Image Preprocessing & Biometrics
# ---------------------------------------------------------------------------

def enhance_illumination(frame_bgr):
    """Enhance illumination using CLAHE on the Lightness channel for shadowed faces."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def calculate_ear(eye_pts):
    """
    Calculate Eye Aspect Ratio (EAR).
    eye_pts: array of 6 points [p1(corner1), p2(top1), p3(top2), p4(corner2), p5(bot2), p6(bot1)]
    """
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    h = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def estimate_head_pose(landmarks_px, img_w, img_h):
    """
    Estimate head pitch, yaw, roll using OpenCV SolvePnP.
    Returns (pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs)
    """
    image_points = np.array([landmarks_px[idx][:2] for idx in HEAD_LANDMARK_INDICES], dtype=np.float64)

    focal_length = float(img_w)
    center = (float(img_w) / 2.0, float(img_h) / 2.0)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        HEAD_MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0, None, None, cam_matrix, dist_coeffs

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    return pitch, yaw, roll, rvec, tvec, cam_matrix, dist_coeffs


def check_head_silhouette_enhanced(frame_bgr, roi_box):
    """
    Evaluates head/upper-head silhouette using Canny edge density, Sobel gradient, 
    HSV color variance, and structural texture analysis.
    Returns (is_head_visible, score)
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = roi_box

    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. Edge density
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.count_nonzero(edges) / max(edges.size, 1)

    # 2. Sobel Gradient Magnitude for head outline contours
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    mean_grad = float(np.mean(grad_mag))

    # 3. Standard deviation of intensity
    std_dev = float(np.std(gray))

    # 4. Color variance in HSV (hair/skin contrast vs uniform background)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))

    # Weighted Silhouette Score
    score = (
        0.35 * min(1.0, edge_density / 0.08) +
        0.25 * min(1.0, mean_grad / 25.0) +
        0.20 * min(1.0, std_dev / 30.0) +
        0.20 * min(1.0, (sat_std + val_std) / 40.0)
    )

    is_head_visible = score >= 0.32
    return is_head_visible, score



def _face_box(pts):
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def _is_duplicate_face(box_a, box_b):
    """True if two face boxes overlap enough to be the same face."""
    ix = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    iy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = ix * iy
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union > 0 and inter / union > 0.30:
        return True
    ca = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    cb = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    min_side = min(box_a[2] - box_a[0], box_a[3] - box_a[1],
                   box_b[2] - box_b[0], box_b[3] - box_b[1])
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < 0.6 * max(min_side, 1.0)


# ---------------------------------------------------------------------------
# Independent Student Tracker
# ---------------------------------------------------------------------------

class StudentTracker:
    """Tracks a single student's biometric state across video frames."""

    def __init__(self, student_id):
        self.student_id = student_id
        self.bbox = (0, 0, 0, 0)
        self.center = (0.0, 0.0)
        self.baseline_y = 0.0
        self.baseline_pitch = 0.0
        self.last_seen_frame = 0

        self.ear = 0.30
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.last_pitch = 0.0
        self.upright_pitch = None      # learned upright head pitch for this student
        self.head_drop = 0.0           # degrees below upright_pitch (smoothed)
        self.head_dropped = False
        self.pitch_history_recent = deque(maxlen=HEAD_SMOOTH_FRAMES)
        self.fast_counter = 0

        self.ear_history = deque(maxlen=PERCLOS_WINDOW_SIZE)
        self.closed_history = deque(maxlen=PERCLOS_WINDOW_SIZE)       # gated by head drop
        self.closed_raw_history = deque(maxlen=PERCLOS_WINDOW_SIZE)   # eyes/posture, ungated
        self.pitch_history = deque(maxlen=PERCLOS_WINDOW_SIZE)

        self.status = "AWAKE"
        self.drowsy_counter = 0
        self.awake_counter = 0
        self.consecutive_face_lost_frames = 0
        self.severe_slouch_counter = 0
        self.is_severe_slouch = False
        self.perclos = 0.0
        self.score = 0.0  # Drowsiness score % (0..100)

    def update(self, bbox, ear, pitch, yaw, roll, frame_id):
        """Update tracker state when facial landmarks are successfully detected."""
        self.bbox = bbox
        cy = (bbox[1] + bbox[3]) / 2.0
        cx = (bbox[0] + bbox[2]) / 2.0
        
        # Smooth position update
        if self.center[0] == 0.0 and self.center[1] == 0.0:
            self.center = (cx, cy)
        else:
            self.center = (0.75 * self.center[0] + 0.25 * cx, 0.75 * self.center[1] + 0.25 * cy)

        self.last_seen_frame = frame_id

        # Update posture Y baseline when upright & open-eyed
        if pitch > -10.0 and ear > 0.22:
            if self.baseline_y == 0.0:
                self.baseline_y = cy
                self.baseline_pitch = pitch
            else:
                self.baseline_y = 0.95 * self.baseline_y + 0.05 * cy
                self.baseline_pitch = 0.95 * self.baseline_pitch + 0.05 * pitch

        self.consecutive_face_lost_frames = 0
        self.severe_slouch_counter = 0
        self.is_severe_slouch = False

        self.ear = ear
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.last_pitch = pitch

        self.ear_history.append(ear)
        self.pitch_history.append(pitch)

        # Head-drop tracking: how far has this student's head sagged from their own
        # upright posture? (the reference only rises while upright, so a student who
        # stays slumped keeps counting as dropped)
        self.pitch_history_recent.append(pitch)
        smooth_pitch = float(np.mean(self.pitch_history_recent))
        if self.upright_pitch is None:
            if len(self.pitch_history) >= HEAD_BASELINE_FRAMES:
                self.upright_pitch = float(np.median(list(self.pitch_history)[:HEAD_BASELINE_FRAMES]))
        else:
            if smooth_pitch > self.upright_pitch - HEAD_DROP_DEG / 2.0:
                self.upright_pitch = 0.98 * self.upright_pitch + 0.02 * smooth_pitch
        if self.upright_pitch is None:
            self.head_drop = 0.0
        else:
            self.head_drop = self.upright_pitch - smooth_pitch
        self.head_dropped = self.head_drop >= HEAD_DROP_DEG

        eyes_count = self.head_dropped or not REQUIRE_HEAD_DROP_FOR_EYES

        # Fast confirmation counter (tolerates single-frame landmark noise / blinks)
        if self.head_dropped and ear < EAR_CLOSED_THRESHOLD:
            self.fast_counter += 1
        else:
            self.fast_counter = max(0, self.fast_counter - 2)
        fast_confirmed = self.fast_counter >= FAST_DROWSY_FRAMES

        # Instantaneous closed/slouching condition
        relative_pitch = pitch - self.baseline_pitch if self.baseline_pitch != 0.0 else pitch
        is_closed_or_slouching = ((ear < EAR_CLOSED_THRESHOLD) and eyes_count) or (relative_pitch < PITCH_DOWN_THRESHOLD)
        self.closed_history.append(is_closed_or_slouching)
        self.closed_raw_history.append((ear < EAR_CLOSED_THRESHOLD) or (relative_pitch < PITCH_DOWN_THRESHOLD))

        # Calculate PERCLOS over window
        # While the head is sagging, the recent eye closure counts retroactively
        # (eyes closed for a while, THEN head dropped = falling asleep). While the head
        # is upright, lowered eyes are ignored (reading / looking down).
        hist = self.closed_raw_history if (self.head_dropped or not REQUIRE_HEAD_DROP_FOR_EYES) else self.closed_history
        self.perclos = sum(hist) / max(len(hist), 1)

        # Drowsiness confidence score calculation
        ear_score = max(0.0, (EAR_OPEN_THRESHOLD - ear) / EAR_OPEN_THRESHOLD) if eyes_count else 0.0
        pitch_score = max(0.0, (-pitch) / 30.0) if pitch < 0 else 0.0
        combined_instant = min(1.0, max(ear_score, pitch_score))
        self.score = float(np.clip((0.6 * self.perclos + 0.4 * combined_instant) * 100.0, 0.0, 100.0))
        if fast_confirmed:
            self.score = max(self.score, SCORE_DROWSY_THRESHOLD)   # no waiting for the average

        # Score-based state machine with hysteresis
        if self.status == "AWAKE":
            if len(self.closed_history) >= MIN_FRAMES_FOR_DECISION and self.score >= SCORE_DROWSY_THRESHOLD:
                self.status = "DROWSY"
        else:  # DROWSY or SEVERE SLOUCH
            if self.score < SCORE_AWAKE_THRESHOLD:
                self.status = "AWAKE"

        return self.status

    def update_fallback_slouch(self, frame_bgr, frame_id):
        """
        Fallback evaluation when face landmarks are lost.
        Checks for downward trajectory + upper-head/hair region silhouette.
        """
        self.consecutive_face_lost_frames += 1
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.bbox
        bw = max(25, x2 - x1)
        bh = max(25, y2 - y1)

        # Expected region for slouched head (shifted downward)
        expected_y1 = max(0, int(y1 + bh * 0.15))
        expected_y2 = min(h, int(y2 + bh * 0.45))
        roi_x1 = max(0, int(x1 - bw * 0.10))
        roi_x2 = min(w, int(x2 + bw * 0.10))
        roi_y1 = expected_y1
        roi_y2 = expected_y2

        expected_cy = (roi_y1 + roi_y2) / 2.0
        downward_shift = (expected_cy - self.baseline_y) if self.baseline_y > 0 else 30.0

        # Check enhanced silhouette in ROI
        is_silhouette_visible, sil_score = check_head_silhouette_enhanced(
            frame_bgr, (roi_x1, roi_y1, roi_x2, roi_y2)
        )

        # Condition: Previously tracked student + significant downward shift + visible head silhouette
        is_slouching_condition = (
            (downward_shift >= SLOUCH_DOWNWARD_SHIFT_PX or self.last_pitch < -10.0)
            and is_silhouette_visible
        )

        if is_slouching_condition:
            self.severe_slouch_counter += 1
        else:
            self.severe_slouch_counter = max(0, self.severe_slouch_counter - 1)

        # Require persistence for several seconds before triggering SEVERE SLOUCH
        if self.severe_slouch_counter >= SLOUCH_PERSISTENCE_FRAMES:
            self.status = "SEVERE SLOUCH"
            self.score = float(min(99.0, 90.0 + sil_score * 9.0))
            self.is_severe_slouch = True
            self.bbox = (roi_x1, roi_y1, roi_x2, roi_y2)
            self.center = ((roi_x1 + roi_x2) / 2.0, expected_cy)
            self.last_seen_frame = frame_id
        else:
            self.is_severe_slouch = False

        return self.status


# ---------------------------------------------------------------------------
# Detector Engine
# ---------------------------------------------------------------------------

class DrowsinessDetector:
    """Main High-Accuracy Classroom Drowsiness Detection Engine."""

    def __init__(self):
        ensure_model()
        self.landmarker = self._make_landmarker()
        # Second landmarker (own state) used by the background discovery scan,
        # because a MediaPipe task object must not be shared between threads.
        self.scan_landmarker = self._make_landmarker()
        self._scan_lock = threading.Lock()
        self._scan_thread = None
        self._scan_results = []
        self._roi_cache = {}   # student_id -> (frame_id, pts_px)

        # OpenCV Haar Cascade fallback detector
        self.cascade_detector = cv2.CascadeClassifier(CASCADE_PATH)

        self.trackers = {}  # student_id -> StudentTracker
        self.next_student_id = 1
        self.frame_count = 0
        self.global_alarm_counter = 0
        self.global_alarm_active = False

    @staticmethod
    def _make_landmarker():
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=15,
            min_face_detection_confidence=0.30,
            min_face_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        )
        return FaceLandmarker.create_from_options(options)

    def close(self):
        """Release both landmarkers (waits briefly for a running scan)."""
        t = self._scan_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        self.landmarker.close()
        self.scan_landmarker.close()

    def _start_background_scan(self, frame_bgr):
        if self._scan_thread is not None and self._scan_thread.is_alive():
            return
        frame_copy = frame_bgr.copy()

        def worker():
            try:
                found = self._scan_tiles(frame_copy, self.scan_landmarker)
            except Exception as exc:   # never let the scanner kill the video
                print(f"Discovery scan failed: {exc}")
                found = []
            with self._scan_lock:
                self._scan_results = found

        self._scan_thread = threading.Thread(target=worker, daemon=True)
        self._scan_thread.start()

    def _collect_scan_results(self):
        with self._scan_lock:
            found, self._scan_results = self._scan_results, []
        return found

    # ------------------------------------------------------------------
    # Face detection helpers (full frame / tracked ROI / tiled scan)
    # ------------------------------------------------------------------
    def _detect_pts(self, bgr, offset=(0.0, 0.0), scale=1.0, landmarker=None):
        """
        Run the landmarker on `bgr` and return a list of (468+, 3) pixel arrays
        expressed in ORIGINAL-frame coordinates (offset = crop origin in the
        original frame, scale = how much the crop was enlarged).
        """
        ih, iw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = (landmarker or self.landmarker).detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
        out = []
        if not result or not result.face_landmarks:
            return out
        for lms in result.face_landmarks:
            pts = np.array([
                [lm.x * iw / scale + offset[0],
                 lm.y * ih / scale + offset[1],
                 lm.z * iw / scale] for lm in lms
            ], dtype=np.float64)
            out.append(pts)
        return out

    def _detect_in_roi(self, frame_bgr, bbox):
        """Re-detect one known face inside an upscaled crop around its last bbox."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        side = max(fw, fh) * (1.0 + 2.0 * ROI_MARGIN)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx1 = int(max(0, cx - side / 2)); rx2 = int(min(w, cx + side / 2))
        ry1 = int(max(0, cy - side / 2)); ry2 = int(min(h, cy + side / 2))
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            return None
        crop = frame_bgr[ry1:ry2, rx1:rx2]
        scale = ROI_INPUT_SIZE / float(max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        for attempt in (crop, enhance_illumination(crop)):   # 2nd try: CLAHE for shadows
            found = self._detect_pts(attempt, offset=(rx1, ry1), scale=scale)
            if found:
                # choose the face closest to where we expect the student to be
                found.sort(key=lambda p: math.hypot(
                    (p[:, 0].min() + p[:, 0].max()) / 2.0 - cx,
                    (p[:, 1].min() + p[:, 1].max()) / 2.0 - cy))
                return found[0]
        return None

    def _scan_tiles(self, frame_bgr, landmarker=None):
        """Multi-scale tiled scan: finds small faces the whole-frame pass misses."""
        h, w = frame_bgr.shape[:2]
        found = []
        for frac in TILE_FRACTIONS:
            tile = int(min(max(128, w * frac), w, h))
            step = max(1, int(tile * (1.0 - TILE_OVERLAP)))
            xs = list(range(0, max(1, w - tile + 1), step))
            ys = list(range(0, max(1, h - tile + 1), step))
            if xs[-1] != w - tile: xs.append(w - tile)
            if ys[-1] != h - tile: ys.append(h - tile)
            scale = TILE_INPUT_SIZE / float(tile)
            margin = 0.02 * tile
            for ty in ys:
                for tx in xs:
                    crop = frame_bgr[ty:ty + tile, tx:tx + tile]
                    crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
                    for pts in self._detect_pts(crop, offset=(tx, ty), scale=scale, landmarker=landmarker):
                        bx1, by1, bx2, by2 = _face_box(pts)
                        bw, bh = bx2 - bx1, by2 - by1
                        if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
                            continue
                        if not (0.5 <= bw / bh <= 1.5):          # implausible shape
                            continue
                        # drop faces cut off by an interior tile border
                        if ((bx1 - tx < margin and tx > 0) or
                                (by1 - ty < margin and ty > 0) or
                                (tx + tile - bx2 < margin and tx + tile < w) or
                                (ty + tile - by2 < margin and ty + tile < h)):
                            continue
                        box = (bx1, by1, bx2, by2)
                        if not any(_is_duplicate_face(box, _face_box(p)) for p in found):
                            found.append(pts)
        return found

    def process_frame(self, frame_bgr):
        self.frame_count += 1
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()

        # Pass A: whole-frame detection (fast; finds large / close faces)
        faces_pts = self._detect_pts(frame_bgr)

        # Pass B: follow KNOWN students on an upscaled crop around their last
        # position. Refreshed every ROI_REFRESH_FRAMES; cached in between.
        for s_id, tracker in list(self.trackers.items()):
            if tracker.bbox == (0, 0, 0, 0):
                continue
            if any(_is_duplicate_face(_face_box(p), tracker.bbox) for p in faces_pts):
                self._roi_cache.pop(s_id, None)
                continue
            cached = self._roi_cache.get(s_id)
            # stagger refreshes per student so the cost is spread evenly across frames
            if cached and self.frame_count - cached[0] < ROI_REFRESH_FRAMES + (s_id % ROI_REFRESH_FRAMES):
                faces_pts.append(cached[1])
                continue
            roi_pts = self._detect_in_roi(frame_bgr, tracker.bbox)
            if roi_pts is not None:
                self._roi_cache[s_id] = (self.frame_count, roi_pts)
                faces_pts.append(roi_pts)
            else:
                self._roi_cache.pop(s_id, None)

        # Pass C: tiled scan to DISCOVER small / distant faces.
        # First frame runs synchronously (instant result); later scans run in a
        # background thread so playback never stalls.
        if self.frame_count == 1:
            discovered = self._scan_tiles(frame_bgr)
        else:
            if self.frame_count % DISCOVERY_INTERVAL_FRAMES == 1:
                self._start_background_scan(frame_bgr)
            discovered = self._collect_scan_results()
        for pts in discovered:
            box = _face_box(pts)
            if not any(_is_duplicate_face(box, _face_box(p)) for p in faces_pts):
                faces_pts.append(pts)

        current_face_data = []

        for pts_px in faces_pts:

            # Bounding box
            min_x, max_x = int(np.min(pts_px[:, 0])), int(np.max(pts_px[:, 0]))
            min_y, max_y = int(np.min(pts_px[:, 1])), int(np.max(pts_px[:, 1]))

            # Padding
            pad_x = int((max_x - min_x) * 0.10)
            pad_y = int((max_y - min_y) * 0.10)
            x1 = max(0, min_x - pad_x)
            y1 = max(0, min_y - pad_y)
            x2 = min(w, max_x + pad_x)
            y2 = min(h, max_y + pad_y)
            bbox = (x1, y1, x2, y2)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            # Biometrics calculation
            left_eye_pts = pts_px[LEFT_EYE_INDICES, :2]
            right_eye_pts = pts_px[RIGHT_EYE_INDICES, :2]
            left_ear = calculate_ear(left_eye_pts)
            right_ear = calculate_ear(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            pitch, yaw, roll, rvec, tvec, cam_mat, dist_coeffs = estimate_head_pose(pts_px, w, h)

            current_face_data.append({
                "bbox": bbox,
                "center": center,
                "ear": ear,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "pts_px": pts_px,
                "rvec": rvec,
                "tvec": tvec,
                "cam_mat": cam_mat,
                "dist_coeffs": dist_coeffs,
                "left_eye": left_eye_pts,
                "right_eye": right_eye_pts
            })

        # Match detected faces with existing trackers using centroid distance
        matched_tracker_ids = set()
        active_student_states = []

        for face in current_face_data:
            cx, cy = face["center"]

            best_id = None
            min_dist = MAX_MATCH_DISTANCE_PX

            for s_id, tracker in self.trackers.items():
                if s_id in matched_tracker_ids:
                    continue
                dist = math.hypot(cx - tracker.center[0], cy - tracker.center[1])
                if dist < min_dist:
                    min_dist = dist
                    best_id = s_id

            if best_id is None:
                best_id = self.next_student_id
                self.next_student_id += 1
                self.trackers[best_id] = StudentTracker(best_id)

            matched_tracker_ids.add(best_id)
            tracker = self.trackers[best_id]

            # Update tracker biometrics
            tracker.update(
                bbox=face["bbox"],
                ear=face["ear"],
                pitch=face["pitch"],
                yaw=face["yaw"],
                roll=face["roll"],
                frame_id=self.frame_count
            )

            # Visual Annotation
            self._annotate_face(display, face, tracker)

            active_student_states.append({
                "id": tracker.student_id,
                "label": f"Student {tracker.student_id}",
                "status": tracker.status,
                "score": float(tracker.score),
                "ear": float(tracker.ear),
                "pitch": float(tracker.pitch),
                "perclos": float(tracker.perclos * 100.0)
            })

        # Evaluate Fallback Severe Slouch for unmatched/lost face trackers
        unmatched_ids = set(self.trackers.keys()) - matched_tracker_ids
        for s_id in unmatched_ids:
            tracker = self.trackers[s_id]
            tracker.update_fallback_slouch(frame_bgr, self.frame_count)

            if tracker.is_severe_slouch:
                self._annotate_severe_slouch(display, tracker)
                active_student_states.append({
                    "id": tracker.student_id,
                    "label": f"Student {tracker.student_id}",
                    "status": "SEVERE SLOUCH",
                    "score": float(tracker.score),
                    "ear": 0.0,
                    "pitch": -35.0,
                    "perclos": 100.0
                })

        # Purge stale trackers not seen recently
        stale_ids = []
        for s_id, tracker in self.trackers.items():
            max_allowed = MAX_SLOUCH_MISSING_FRAMES if tracker.is_severe_slouch else MAX_TRACKER_MISSING_FRAMES
            if self.frame_count - tracker.last_seen_frame > max_allowed:
                stale_ids.append(s_id)

        for s_id in stale_ids:
            del self.trackers[s_id]
            self._roi_cache.pop(s_id, None)

        # Aggregate Classroom Drowsiness Statistics
        detected_count = len(active_student_states)
        drowsy_count = sum(1 for s in active_student_states if s["status"] in ("DROWSY", "SEVERE SLOUCH"))
        awake_count = detected_count - drowsy_count
        percentage = (drowsy_count / detected_count * 100.0) if detected_count > 0 else 0.0
        
        is_above_threshold = detected_count > 0 and (percentage / 100.0) >= ALARM_THRESHOLD_PERCENT
        if is_above_threshold:
            self.global_alarm_counter += 1
        else:
            self.global_alarm_counter = max(0, self.global_alarm_counter - 2)
            
        if self.global_alarm_counter >= ALARM_PERSISTENCE_FRAMES:
            self.global_alarm_active = True
        elif self.global_alarm_counter == 0:
            self.global_alarm_active = False

        # Draw main classroom stats overlay
        self._draw_global_overlay(display, detected_count, drowsy_count, percentage, self.global_alarm_active)

        return {
            "detected": detected_count,
            "drowsy": drowsy_count,
            "awake": awake_count,
            "percentage": percentage,
            "alarm": self.global_alarm_active,
            "faces": active_student_states,
            "frame": display
        }

    def _annotate_face(self, display, face, tracker):
        x1, y1, x2, y2 = face["bbox"]
        is_drowsy = (tracker.status == "DROWSY")

        color = (0, 0, 255) if is_drowsy else (0, 200, 0)
        thickness = 2

        # Face Bounding Box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

        # Draw Eye Contours
        for pt in face["left_eye"].astype(int):
            cv2.circle(display, tuple(pt), 2, (255, 255, 0), -1)
        for pt in face["right_eye"].astype(int):
            cv2.circle(display, tuple(pt), 2, (255, 255, 0), -1)

        # Draw Nose Direction Pointer Line from Pose
        if face["rvec"] is not None and face["tvec"] is not None:
            nose_end_3d = np.array([[0.0, 0.0, 150.0]], dtype=np.float64)
            nose_end_2d, _ = cv2.projectPoints(
                nose_end_3d, face["rvec"], face["tvec"], face["cam_mat"], face["dist_coeffs"]
            )
            p1 = (int(face["pts_px"][1][0]), int(face["pts_px"][1][1]))
            p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            cv2.line(display, p1, p2, (255, 100, 0), 2)

        # Student Label Header & Metrics Badge
        label_head = f"S{tracker.student_id}: {tracker.status} ({tracker.score:.0f}%)"
        label_sub = f"EAR:{tracker.ear:.2f} Pitch:{tracker.pitch:.0f} deg"

        # Background badge for text
        cv2.rectangle(display, (x1, max(0, y1 - 32)), (x1 + 180, y1), color, -1)
        cv2.putText(
            display, label_head, (x1 + 4, max(12, y1 - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            display, label_sub, (x1 + 4, max(24, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA
        )

    def _annotate_severe_slouch(self, display, tracker):
        x1, y1, x2, y2 = tracker.bbox
        color = (0, 120, 255)  # Orange / Amber for Severe Slouch

        # Upper Head Region Box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Downward trajectory arrow
        cx = int((x1 + x2) / 2)
        cv2.arrowedLine(display, (cx, y1 - 10), (cx, y1 + 15), color, 2, tipLength=0.4)

        # Debug Badge Label
        label_head = f"S{tracker.student_id}: SEVERE SLOUCH -> DROWSY"
        label_sub = "Face Lost | Upper Head Silhouette Active"

        cv2.rectangle(display, (x1, max(0, y1 - 34)), (x1 + 250, y1), color, -1)
        cv2.putText(
            display, label_head, (x1 + 4, max(12, y1 - 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            display, label_sub, (x1 + 4, max(24, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1, cv2.LINE_AA
        )

    def _draw_global_overlay(self, display, detected, drowsy, percentage, alarm_active):
        h, w = display.shape[:2]

        # Status Bar at bottom
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (12, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

        bar_text = (
            f"Detected: {detected}   |   Drowsy: {drowsy}   |   "
            f"Awake: {detected - drowsy}   |   Ratio: {percentage:.1f}%"
        )
        cv2.putText(
            display, bar_text, (16, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 220, 255), 2, cv2.LINE_AA
        )

        # Global Alarm Banner
        if alarm_active:
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            banner = display.copy()
            cv2.rectangle(banner, (0, 0), (w, 48), (0, 0, 180), -1)
            cv2.addWeighted(banner, 0.70, display, 0.30, 0, display)
            cv2.putText(
                display, "ALERT: CLASSROOM DROWSINESS EXCEEDS 40%!",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 230, 80), 2, cv2.LINE_AA
            )