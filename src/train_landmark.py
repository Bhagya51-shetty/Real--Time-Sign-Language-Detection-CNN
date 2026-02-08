
# for training the landmark of our data we used the dataset path  and some features 

# with spliting the data by X, Y, Z etc

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ✅ Load the landmark dataset
data = np.load("landmarks_data.npz")
X, y = data["X"], data["y"]
print("✅ Data loaded:", X.shape, y.shape)

# ✅ Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "landmarks_scaler.pkl")
print("✅ Scaler saved -> landmarks_scaler.pkl")

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),                # 21 hand points × 3 coords
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=25, batch_size=32, verbose=1)

# ✅ Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")

# ✅ Save model in new format
model.save("C:/sign-bridge-main/sign-bridge-main/landmarks_model.keras", save_format="keras")
print("\n✅ Model saved successfully → landmarks_model.keras")
