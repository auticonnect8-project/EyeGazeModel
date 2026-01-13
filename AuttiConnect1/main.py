import cv2
import mediapipe as mp
import time
from gtts import gTTS
import os
from playsound import playsound
from collections import deque

# =========================
# Text to Speech Function
# =========================
def speak_warning():
    text = "Please look at the screen"
    tts = gTTS(text=text, lang='en')
    file = "warning.mp3"
    tts.save(file)
    playsound(file)
    os.remove(file)

# =========================
# MediaPipe Face Mesh Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Webcam Setup
# =========================
cap = cv2.VideoCapture(0)
not_looking_start = None
WARNING_TIME = 10  # seconds

# =========================
# Eye Position History for Smoothing
# =========================
eye_history = deque(maxlen=5)  # average over last 5 frames

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking_at_screen = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        # Left and Right eye landmarks
        left_eye_left = landmarks.landmark[33]
        left_eye_right = landmarks.landmark[133]
        right_eye_left = landmarks.landmark[362]
        right_eye_right = landmarks.landmark[263]

        # Convert to pixel coordinates
        left_eye_center_x = int((left_eye_left.x + left_eye_right.x) / 2 * w)
        left_eye_center_y = int((left_eye_left.y + left_eye_right.y) / 2 * h)
        right_eye_center_x = int((right_eye_left.x + right_eye_right.x) / 2 * w)
        right_eye_center_y = int((right_eye_left.y + right_eye_right.y) / 2 * h)

        # Draw eye centers for debugging
        cv2.circle(frame, (left_eye_center_x, left_eye_center_y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye_center_x, right_eye_center_y), 3, (0, 255, 0), -1)

        # Normalized eye center (0 = left, 1 = right)
        norm_x = ((left_eye_center_x + right_eye_center_x) / 2) / w - 0.5
        norm_y = ((left_eye_center_y + right_eye_center_y) / 2) / h - 0.5

        # Add to history for smoothing
        eye_history.append((norm_x, norm_y))

        # Compute smoothed values
        avg_x = sum([e[0] for e in eye_history]) / len(eye_history)
        avg_y = sum([e[1] for e in eye_history]) / len(eye_history)

        # Thresholds for looking at screen
        if abs(avg_x) < 0.15 and abs(avg_y) < 0.15:
            looking_at_screen = True
        else:
            looking_at_screen = False

        # Display normalized values
        cv2.putText(frame, f"X: {avg_x:.2f}, Y: {avg_y:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        # No face detected
        looking_at_screen = False

    # =========================
    # Attention Timer Logic
    # =========================
    if looking_at_screen:
        not_looking_start = None
        status = "LOOKING AT SCREEN"
        color = (0, 255, 0)
    else:
        if not_looking_start is None:
            not_looking_start = time.time()

        elapsed = time.time() - not_looking_start

        if elapsed >= WARNING_TIME:
            speak_warning()
            not_looking_start = time.time()  # reset timer after warning

        status = f"NOT LOOKING ({int(elapsed)}s)"
        color = (0, 0, 255)

    # =========================
    # Display Status
    # =========================
    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Attention Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
