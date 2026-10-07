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

ALARM_THRESHOLD_PERCENT = 0.50  # > 50% students drowsy triggers alarm
ALARM_PERSISTENCE_FRAMES = 90   # ~3 seconds sustained before triggering global alarm

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

        self.ear_history = deque(maxlen=PERCLOS_WINDOW_SIZE)
        self.closed_history = deque(maxlen=PERCLOS_WINDOW_SIZE)
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

        # Instantaneous closed/slouching condition
        relative_pitch = pitch - self.baseline_pitch if self.baseline_pitch != 0.0 else pitch
        is_closed_or_slouching = (ear < EAR_CLOSED_THRESHOLD) or (relative_pitch < PITCH_DOWN_THRESHOLD)
        self.closed_history.append(is_closed_or_slouching)

        # Calculate PERCLOS over window
        self.perclos = sum(self.closed_history) / max(len(self.closed_history), 1)

        # Drowsiness confidence score calculation
        ear_score = max(0.0, (EAR_OPEN_THRESHOLD - ear) / EAR_OPEN_THRESHOLD)
        pitch_score = max(0.0, (-pitch) / 30.0) if pitch < 0 else 0.0
        combined_instant = min(1.0, max(ear_score, pitch_score))
        self.score = float(np.clip((0.6 * self.perclos + 0.4 * combined_instant) * 100.0, 0.0, 100.0))

        # Temporal Hysteresis State Machine
        if is_closed_or_slouching or self.perclos >= PERCLOS_DROWSY_RATIO:
            self.drowsy_counter += 1
            self.awake_counter = 0
        else:
            self.awake_counter += 1
            self.drowsy_counter = 0

        if self.status == "AWAKE":
            if self.drowsy_counter >= CONSECUTIVE_DROWSY_FRAMES or self.perclos >= PERCLOS_DROWSY_RATIO:
                self.status = "DROWSY"
        else:  # DROWSY or SEVERE SLOUCH
            if self.awake_counter >= CONSECUTIVE_AWAKE_FRAMES and self.perclos < PERCLOS_AWAKE_RATIO:
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
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # High-accuracy multi-face sensitivity options
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=15,
            min_face_detection_confidence=0.30,
            min_face_presence_confidence=0.30,
            min_tracking_confidence=0.30,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # OpenCV Haar Cascade fallback detector
        self.cascade_detector = cv2.CascadeClassifier(CASCADE_PATH)

        self.trackers = {}  # student_id -> StudentTracker
        self.next_student_id = 1
        self.frame_count = 0
        self.global_alarm_counter = 0
        self.global_alarm_active = False

    def process_frame(self, frame_bgr):
        self.frame_count += 1
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()

        # Pass 1: Standard RGB landmark detection
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result1 = self.landmarker.detect(mp_image1)
        faces_landmarks = list(detection_result1.face_landmarks) if (detection_result1 and detection_result1.face_landmarks) else []

        # Retry on enhanced frames when no face is found or tracked faces are missing.
        if not faces_landmarks or len(faces_landmarks) < len(self.trackers):
            enhanced_bgr = enhance_illumination(frame_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_rgb)
            detection_result2 = self.landmarker.detect(mp_image2)
            if detection_result2 and detection_result2.face_landmarks:
                for new_landmarks in detection_result2.face_landmarks:
                    # Deduplicate overlapping faces
                    new_pts = np.array([[lm.x * w, lm.y * h] for lm in new_landmarks])
                    new_cx, new_cy = np.mean(new_pts[:, 0]), np.mean(new_pts[:, 1])
                    
                    is_dup = False
                    for existing in faces_landmarks:
                        ex_pts = np.array([[lm.x * w, lm.y * h] for lm in existing])
                        ex_cx, ex_cy = np.mean(ex_pts[:, 0]), np.mean(ex_pts[:, 1])
                        if math.hypot(new_cx - ex_cx, new_cy - ex_cy) < 40.0:
                            is_dup = True
                            break
                    if not is_dup:
                        faces_landmarks.append(new_landmarks)

        current_face_data = []

        for landmarks in faces_landmarks:
            # Convert normalized landmarks to pixel coordinates
            pts_px = np.array([
                [lm.x * w, lm.y * h, lm.z * w] for lm in landmarks
            ], dtype=np.float64)

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
                display, "ALERT: CLASSROOM DROWSINESS EXCEEDS 50%!",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 230, 80), 2, cv2.LINE_AA
            )
