"""
Generate landmarks (x,y,z for each of 21 hand keypoints = 63 features)
from a directory-based dataset of ASL images.

Expected folder structure:
DATA_DIR/
    A/
       img1.jpg
       img2.jpg
    B/
       ...
"""

import os
import glob
import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm

# ✅ Use raw string to prevent escape sequences
DATA_DIR = r"C:\sign-bridge-main\sign-bridge-main\asl_alphabet_train\asl_alphabet_train"
OUT_FILE = "landmarks_data.npz"

# ✅ Check DATA_DIR exists
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"❌ DATA_DIR not found: {DATA_DIR}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X, y = [], []

# ✅ Safely get class folders
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print("✅ Detected classes:", class_names)

for label_idx, cls in enumerate(class_names):
    cls_path = os.path.join(DATA_DIR, cls)
    images = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))

    for img_path in tqdm(images, desc=f"Processing {cls}"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            landmark_row = []
            for lm in hand.landmark:
                landmark_row.extend([lm.x, lm.y, lm.z])

            X.append(landmark_row)
            y.append(label_idx)

hands.close()

X = np.array(X)
y = np.array(y)
print("✅ Final landmark dataset shape:", X.shape, y.shape)

np.savez_compressed(OUT_FILE, X=X, y=y)
np.save("landmarks_classes.npy", np.array(class_names))

print(f"\n🎉 Successfully saved:")
print(f"• Landmark data → {OUT_FILE}")
print(f"• Class names → landmarks_classes.npy\n")
