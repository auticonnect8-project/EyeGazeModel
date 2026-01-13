import pygame
import cv2
import mediapipe as mp
import time
from gtts import gTTS
from playsound import playsound
from collections import deque
import threading
import os

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
eye_history = deque(maxlen=5)  # smoothing

# =========================
# Function to check attention in background
# =========================
looking_at_screen = True  # global variable

def attention_thread():
    global not_looking_start, looking_at_screen
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Eye landmarks
            left_eye_left = landmarks.landmark[33]
            left_eye_right = landmarks.landmark[133]
            right_eye_left = landmarks.landmark[362]
            right_eye_right = landmarks.landmark[263]

            # Eye center
            left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
            left_eye_center_y = (left_eye_left.y + left_eye_right.y) / 2
            right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2
            right_eye_center_y = (right_eye_left.y + right_eye_right.y) / 2

            # Normalized center
            norm_x = ((left_eye_center_x + right_eye_center_x) / 2) - 0.5
            norm_y = ((left_eye_center_y + right_eye_center_y) / 2) - 0.5

            eye_history.append((norm_x, norm_y))
            avg_x = sum([e[0] for e in eye_history]) / len(eye_history)
            avg_y = sum([e[1] for e in eye_history]) / len(eye_history)

            if abs(avg_x) < 0.15 and abs(avg_y) < 0.15:
                looking_at_screen = True
                not_looking_start = None
            else:
                looking_at_screen = False
        else:
            looking_at_screen = False

        # Timer for warning
        if not looking_at_screen:
            if not_looking_start is None:
                not_looking_start = time.time()
            elapsed = time.time() - not_looking_start
            if elapsed >= WARNING_TIME:
                speak_warning()
                not_looking_start = time.time()

# =========================
# Start attention thread
# =========================
threading.Thread(target=attention_thread, daemon=True).start()

# =========================
# Simple Dodge Game
# =========================
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dodge the Block!")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Player
player_size = 50
player_x = WIDTH // 2
player_y = HEIGHT - player_size - 10
player_speed = 7

# Block
block_width = 50
block_height = 50
block_x = WIDTH // 2
block_y = -50
block_speed = 5

score = 0
running = True

while running:
    clock.tick(60)
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x - player_speed > 0:
        player_x -= player_speed
    if keys[pygame.K_RIGHT] and player_x + player_speed + player_size < WIDTH:
        player_x += player_speed

    # Move block
    block_y += block_speed
    if block_y > HEIGHT:
        block_y = -block_height
        block_x = pygame.mouse.get_pos()[0]  # random or follow mouse
        score += 1

    # Draw player and block
    pygame.draw.rect(screen, (0, 255, 0), (player_x, player_y, player_size, player_size))
    pygame.draw.rect(screen, (255, 0, 0), (block_x, block_y, block_width, block_height))

    # Show attention status
    status_text = "LOOKING" if looking_at_screen else "NOT LOOKING"
    color = (0, 255, 0) if looking_at_screen else (255, 0, 0)
    status_surf = font.render(status_text, True, color)
    screen.blit(status_surf, (10, 10))

    # Show score
    score_surf = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_surf, (WIDTH - 150, 10))

    pygame.display.flip()

pygame.quit()
cap.release()
