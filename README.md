# Classroom Drowsiness Detector

A Flask dashboard for monitoring classroom drowsiness from a webcam or an uploaded video.

## Detection pipeline
- Uses MediaPipe Face Landmarker to detect and track multiple faces.
- Estimates eye openness with EAR and drowsiness over time with PERCLOS.
- Uses head-pose pitch and slouch detection, including severe-slouch fallback.
- Applies temporal smoothing and maintains per-student tracking and scores.
- Shows annotated video, student details, classroom statistics, and alarm status.
- Uses the same `DrowsinessDetector.process_frame()` pipeline for webcam and uploaded-video frames.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11, 64-bit recommended
- Webcam for Webcam mode
- Internet access on first run to download the MediaPipe Face Landmarker asset

## Requirements
- Windows 10/11
- Python 3.10 or 3.11, 64-bit recommended
- Webcam
- Internet access for the first model download

## Installation

Open PowerShell or Command Prompt in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
python app.py
```

The dashboard opens at <http://127.0.0.1:5000>. Keep the terminal open while using it. The first run downloads `models/face_landmarker.task` automatically.

## Dashboard modes

Use **Webcam** for the live camera feed, or select **Upload Video** to choose a video file. MP4, AVI, MOV, MKV, and WebM are supported when OpenCV can decode the file.

Uploaded videos are read and processed frame-by-frame; the entire video is not loaded into memory. The annotated frames and student statistics update progressively using the same detection pipeline as Webcam mode.

Upload a file, then use **Start**, **Pause**, **Resume**, **Stop**, and **Restart** to control playback. Progress and the current/total timestamp are shown below the video. Stop preserves the uploaded file so it can be restarted; uploading another file replaces it.

## Important
This is a classroom/student project and not a medical or safety-certified system. The underlying model was trained on a relatively small research dataset and its labels are described by the model author as weak/heuristic.

## Webcam dashboard controls
- **Snapshot** downloads the current webcam frame.
- **Reset Alarm** resets the webcam alarm state.
- **Allow Alerts** enables browser notifications for alarm events.

The standalone OpenCV webcam view can also be run with `python main.py`. In that view, `Q` quits, `S` saves a screenshot, and `R` resets the alarm.

## Alarm rule
If:
`drowsy_students / detected_students >= 0.50`

for the configured persistence period, the classroom alarm activates.

Example:
- 4 students detected, 3 drowsy = 75% → ALARM
- 4 students detected, 2 drowsy = 50% → ALARM
- 5 students detected, 2 drowsy = 40% → NO ALARM
