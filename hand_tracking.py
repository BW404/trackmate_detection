import cv2
from mediapipe.python import solutions as mp_solutions

mp_hands = mp_solutions.hands
mp_draw = mp_solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,      # each frame independent
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

last_landmarks = None
no_hand_frames = 0
MAX_NO_HAND_FRAMES = 5

def process_frame(frame):
    """
    frame: numpy array (BGR)
    returns processed frame (BGR) with landmarks drawn
    """
    global last_landmarks, no_hand_frames

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
