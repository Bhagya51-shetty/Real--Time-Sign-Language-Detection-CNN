#!/usr/bin/env python3
""" ASL: NO CONTINUOUS SAME LETTER | DAD OK | OO, DD, BOOK BLOCKED IF OO TOGETHER """

import os, joblib, numpy as np, cv2, mediapipe as mp, tensorflow as tf
from collections import deque
import pyttsx3
import threading
import time

# ===================== CONFIG =====================
MODEL_PATH = "landmarks_model.keras"
SCALER_PATH = "landmarks_scaler.pkl"
CLASSES_PATH = "landmarks_classes.npy"
CONF_THRESHOLD = 0.75
SMOOTH_WINDOW = 10
MIN_SIGN_DURATION = 0.7
PAUSE_THRESHOLD = 3.0
MIRROR_PREVIEW = True

# ===================== TTS =====================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_engine.setProperty('volume', 1.0)
tts_engine.startLoop(False)

speech_queue = deque()
sentence_lock = threading.Lock()

# Sentence state
current_word = []
sentence_parts = []
last_hand_time = time.time()
current_sign = None
sign_start_time = 0
last_letter = None  # Track last added letter to block continuation

def speak_full(sentence):
    if sentence.strip():
        full = sentence.strip()
        print(f"\nSENTENCE: {full}")
        with sentence_lock:
            speech_queue.append(full)

# ====================================================
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
classes = np.load(CLASSES_PATH, allow_pickle=True)

SPACE_CLASS = "space"
DEL_CLASS = "del"

# Helpers
def normalize_row_from_landmarks(lm_list):
    arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
    wrist = arr[0]
    arr -= wrist
    span = max(arr[:,0].ptp(), arr[:,1].ptp(), 1e-6)
    arr[:,:2] /= span
    arr[:,2] /= span
    return arr.flatten()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.75, min_tracking_confidence=0.75)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam not opening!")

pred_buffer = deque(maxlen=SMOOTH_WINDOW)

print("\nASL: NO CONTINUOUS SAME LETTER")
print("DAD = OK | BOOK = B O K (OO blocked) | BOYO = OK")
print("Same letter allowed only if different letter in between!\n")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    proc = frame.copy()
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    display_label = "No Hand"
    display_prob = 0.0

    # TTS
    with sentence_lock:
        while speech_queue:
            t = speech_queue.popleft()
            tts_engine.say(t)
    tts_engine.iterate()

    current_time = time.time()

    if res.multi_hand_landmarks:
        last_hand_time = current_time
        lm = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(proc, lm, mp_hands.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                               mp_draw.DrawingSpec(color=(0,0,255), thickness=2))

        feat = normalize_row_from_landmarks(lm.landmark)
        feat_s = scaler.transform(feat.reshape(1,-1)) if scaler else feat.reshape(1,-1)
        probs = model.predict(feat_s, verbose=0)[0]
        pred_buffer.append(probs)
        avg = np.mean(pred_buffer, axis=0)
        idx = np.argmax(avg)
        prob = avg[idx]
        predicted = str(classes[idx])

        if prob >= CONF_THRESHOLD:
            display_label = predicted
            display_prob = prob

            # New sign start
            if predicted != current_sign:
                sign_start_time = current_time
                current_sign = predicted

            # Add after duration
            if (current_time - sign_start_time) >= MIN_SIGN_DURATION:
                added = False

                # SPACE
                if predicted == SPACE_CLASS:
                    if current_word:
                        word = ''.join(current_word).lower()
                        if word: sentence_parts.append(word)
                        current_word = []
                        last_letter = None  # Reset for new word
                    added = True
                    display_label = "SPACE"

                # DELETE
                elif predicted == DEL_CLASS:
                    if current_word:
                        current_word.pop()
                        last_letter = current_word[-1] if current_word else None
                    elif sentence_parts:
                        sentence_parts.pop()
                        last_letter = None
                    added = True
                    display_label = "DEL"

                # LETTER: BLOCK CONTINUOUS SAME
                elif predicted in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    # Allow only if NOT same as last letter
                    if last_letter is None or predicted != last_letter:
                        current_word.append(predicted)
                        last_letter = predicted
                        added = True

                if added:
                    current_sign = None
                    sign_start_time = current_time

    else:
        # Pause → speak
        if current_time - last_hand_time > PAUSE_THRESHOLD and (sentence_parts or current_word):
            final = ''.join(current_word).lower()
            if final: sentence_parts.append(final)
            sentence = " ".join(sentence_parts)
            speak_full(sentence)
            sentence_parts = []
            current_word = []
            last_letter = None
            current_sign = None

    # Preview
    preview_parts = sentence_parts + ([''.join(current_word).lower()] if current_word else [])
    current_sentence = " ".join(preview_parts)

    if res.multi_hand_landmarks:
        h, w = proc.shape[:2]
        xs = [int(p.x * w) for p in lm.landmark]
        ys = [int(p.y * h) for p in lm.landmark]
        x1, y1 = max(min(xs)-40, 0), max(min(ys)-40, 0)
        x2, y2 = min(max(xs)+40, w), min(max(ys)+40, h)
        cv2.rectangle(proc, (x1,y1), (x2,y2), (0,255,0), 3)

    preview = cv2.flip(proc, 1) if MIRROR_PREVIEW else proc
    cv2.putText(preview, f"{display_label} ({display_prob*100:.1f}%)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    cv2.putText(preview, f"→ {current_sentence}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)

    cv2.imshow("ASL: NO OO DD PP", preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
tts_engine.endLoop()
cap.release()
cv2.destroyAllWindows()
