import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # ✅ Use for HTTP streaming
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_draw = mp.solutions.drawing_utils

last_landmarks = None
no_hand_frames = 0
MAX_NO_HAND_FRAMES = 5

def process_frame(frame, run_detection=True):
    global last_landmarks, no_hand_frames

    if run_detection:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            last_landmarks = results.multi_hand_landmarks
            no_hand_frames = 0
        else:
            no_hand_frames += 1
            if no_hand_frames >= MAX_NO_HAND_FRAMES:
                last_landmarks = None

    if last_landmarks:
        for hand_landmarks in last_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return frame
