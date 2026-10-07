# Classroom Drowsiness Detection — PyTorch-Free ONNX Version

This version is designed for Windows machines where PyTorch is blocked by an Application Control policy.

## What it does
- Uses a pretrained ONNX CNN drowsiness model.
- Uses OpenCV's pretrained Haar face detector to find multiple students.
- Classifies every detected face as DROWSY or AWAKE.
- Sounds a Windows alarm when **more than 50%** of detected students are drowsy.
- Shows the live percentage and per-student result.
- Uses CPU inference through ONNX Runtime.
- Does **not** install or import PyTorch, Ultralytics, or TensorFlow.

The pretrained drowsiness model is:
`notgoodkeeper/cnn-based-drowsiness-detection`

Model page:
https://huggingface.co/notgoodkeeper/cnn-based-drowsiness-detection

The model card documents a 412x412x3 ONNX input and a drowsiness-score regression output.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11, 64-bit recommended
- Webcam
- Internet access for the first model download

## Installation

Open Command Prompt in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then run:

```bat
run_windows.bat
```

The first run downloads `models/drowsiness_model.onnx` automatically.

## Important
This is a classroom/student project and not a medical or safety-certified system. The underlying model was trained on a relatively small research dataset and its labels are described by the model author as weak/heuristic.

## Controls
- `Q` = quit
- `S` = save a screenshot
- `R` = reset alarm state

## Alarm rule
If:
`drowsy_students / detected_students > 0.50`

then the alarm activates.

Example:
- 4 students detected, 3 drowsy = 75% → ALARM
- 4 students detected, 2 drowsy = 50% → NO ALARM
- 5 students detected, 3 drowsy = 60% → ALARM
