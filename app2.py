#!/usr/bin/env python3
from flask import Flask, render_template, jsonify
import cv2, os, mediapipe as mp, numpy as np, tensorflow as tf, joblib, threading, time
from collections import deque
import pyttsx3
import queue

app = Flask(__name__)

# ===================== LOAD MODEL =====================
MODEL_PATH = "landmarks_model.keras"
SCALER_PATH = "landmarks_scaler.pkl"
CLASSES_PATH = "landmarks_classes.npy"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
classes = np.load(CLASSES_PATH, allow_pickle=True)

# ===================== CONFIG =====================
CONF_THRESHOLD = 0.75
SMOOTH_WINDOW = 10
MIN_SIGN_DURATION = 0.7
PAUSE_THRESHOLD = 3.0
SPACE_CLASS = "space"
DEL_CLASS = "del"

# ===================== NEW: SAME LETTER REPEAT LOGIC =====================
REPEAT_ALLOW_DELAY = 4.0          # 4 seconds wait for same letter again
last_same_letter_time = 0.0       # tracks when last same letter was added

# ===================== ULTRA STABLE & FAST TTS =====================
tts_queue = queue.Queue()
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None: break
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 1.0)
        try:
            print(f"SPEAKING: {text}")
            engine.say(text)
            engine.runAndWait()
        except: pass
        finally:
            engine.stop()
            del engine
threading.Thread(target=tts_worker, daemon=True).start()

def speak_full(text):
    if text and text.strip():
        tts_queue.put(text.strip() + ".")

# ===================== OPTIMIZED MEDIAPIPE HANDS =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0  # Fastest
)

# ===================== GLOBAL STATE =====================
current_display = "Click 'Start Detection' to begin..."
current_predicted_letter = "No Hand"
is_running = False
cap = None
current_word = []
sentence_parts = []
pred_buffer = deque(maxlen=SMOOTH_WINDOW)
last_hand_time = time.time()
current_sign = None
sign_start_time = 0

def normalize_landmarks(lm_list):
    arr = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
    wrist = arr[0]
    arr -= wrist
    span = max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1]), 1e-6)
    arr[:, :2] /= span
    arr[:, 2] /= span
    return arr.flatten()

def reset_state():
    global current_word, sentence_parts, pred_buffer, last_hand_time, current_sign, sign_start_time
    global last_same_letter_time
    current_word.clear()
    sentence_parts.clear()
    pred_buffer.clear()
    last_hand_time = time.time()
    current_sign = None
    sign_start_time = 0
    last_same_letter_time = 0.0

def detection_loop():
    global is_running, cap, current_display, current_predicted_letter
    global last_hand_time, current_sign, sign_start_time, last_same_letter_time

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        for i in range(1, 5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened(): break
        else:
            current_display = "Camera nahi mil raha!"
            is_running = False
            return

    print("Camera ON - Super Fast Mode!")
    current_display = "Start signing..."
    reset_state()

    while is_running:
        success, frame = cap.read()
        if not success: continue

        frame = cv2.resize(frame, (640, 480))
        proc = frame.copy()
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        current_time = time.time()
        current_predicted_letter = "No Hand"

        if results.multi_hand_landmarks:
            last_hand_time = current_time
            lm = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(proc, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

            feat = normalize_landmarks(lm.landmark)
            feat_s = scaler.transform(feat.reshape(1, -1)) if scaler else feat.reshape(1, -1)
            probs = model.predict(feat_s, verbose=0)[0]
            pred_buffer.append(probs)
            avg = np.mean(pred_buffer, axis=0)
            idx = np.argmax(avg)
            conf = avg[idx]
            predicted = str(classes[idx])

            if conf >= CONF_THRESHOLD:
                current_predicted_letter = predicted.upper()
                if predicted == SPACE_CLASS:
                    current_predicted_letter = "SPACE"
                elif predicted == DEL_CLASS:
                    current_predicted_letter = "DELETE"

                if predicted != current_sign:
                    sign_start_time = current_time
                    current_sign = predicted

                if (current_time - sign_start_time) >= MIN_SIGN_DURATION:
                    added = False

                    if predicted == SPACE_CLASS and current_word:
                        sentence_parts.append(''.join(current_word).lower())
                        current_word.clear()
                        added = True

                    elif predicted == DEL_CLASS:
                        if current_word:
                            current_word.pop()
                        elif sentence_parts:
                            sentence_parts.pop()
                        added = True

                    elif predicted in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        current_time_now = time.time()
                        # Agar pehli baar ya alag letter → direct add
                        if not current_word or predicted != current_word[-1]:
                            current_word.append(predicted)
                            last_same_letter_time = current_time_now
                            added = True
                        # Same letter → 4 second ka rule
                        else:
                            if (current_time_now - last_same_letter_time) >= REPEAT_ALLOW_DELAY:
                                current_word.append(predicted)
                                last_same_letter_time = current_time_now
                                added = True
                                print(f"[REPEAT ACCEPTED] {predicted} after {REPEAT_ALLOW_DELAY}s")

                    if added:
                        current_sign = None

            # Bounding box
            h, w = proc.shape[:2]
            xs = [int(l.x * w) for l in lm.landmark[:21]]
            ys = [int(l.y * h) for l in lm.landmark[:21]]
            x1, y1 = max(min(xs)-30, 0), max(min(ys)-30, 0)
            x2, y2 = min(max(xs)+30, w), min(max(ys)+30, h)
            cv2.rectangle(proc, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Pause detection → sentence complete
        if (current_time - last_hand_time) > PAUSE_THRESHOLD:
            if current_word or sentence_parts:
                if current_word:
                    sentence_parts.append(''.join(current_word).lower())
                    current_word.clear()
                if sentence_parts:
                    sentence = " ".join(sentence_parts)
                    speak_full(sentence)
                    current_display = f"Spoken: {sentence}"
                    reset_state()

        live = " ".join(sentence_parts + ([''.join(current_word).lower()] if current_word else []))
        if live:
            current_display = live

        preview = cv2.flip(proc, 1)
        cv2.putText(preview, f"Letter: {current_predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(preview, current_display, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)
        cv2.imshow("ISL Translator - Ultra Fast", preview)

        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    is_running = False
    if cap: cap.release()
    cap = None
    cv2.destroyAllWindows()
    tts_queue.put(None)

# ===================== ROUTES =====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/community')
def community():
    return render_template('community.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global is_running, cap
    if is_running:
        return jsonify({"status": "already running"})
    if cap: cap.release()
    cap = None
    time.sleep(0.3)
    reset_state()
    is_running = True
    threading.Thread(target=detection_loop, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/status')
def status():
    return jsonify({
        "current_sentence": current_display,
        "current_letter": current_predicted_letter
    })

if __name__ == '__main__':
    app.run(debug=False, threaded=True, port=5000)