import os
import time
import cv2

from drowsiness_core import DrowsinessDetector, ensure_model


def play_alarm():
    """Play alert sound on Windows."""
    try:
        import winsound
        winsound.Beep(1200, 180)
        winsound.Beep(900, 180)
    except Exception:
        pass


def main():
    print("Initializing Classroom Drowsiness Detector (MediaPipe + EAR + PERCLOS + SolvePnP)...")
    ensure_model()
    detector = DrowsinessDetector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    alarm_active = False
    last_alarm = 0.0

    print("\nRunning Classroom Drowsiness Detection Pipeline.")
    print("Press Q to quit, S to save a screenshot, R to reset alarm.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break

            res = detector.process_frame(frame)
            display = res["frame"]
            alarm_should_be_on = res["alarm"]

            if alarm_should_be_on:
                now = time.time()
                if (not alarm_active) or (now - last_alarm > 2.0):
                    play_alarm()
                    last_alarm = now
                alarm_active = True
            else:
                alarm_active = False

            cv2.imshow("Classroom Drowsiness Detector", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                print("Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                filename = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, display)
                print(f"Saved snapshot to {filename}")
            elif key == ord('r') or key == ord('R'):
                alarm_active = False
                print("Alarm state reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
