import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Config
FEATURES_DIR = "features"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("RESUME CLASSIFICATION - MODEL TRAINING")
print("="*60)

# ============================================================
# LOAD FEATURES (from Step 2)
# ============================================================
print("\n[Loading Features from Step 2]")

try:
    # Load features
    features = np.load(f"{FEATURES_DIR}/features.npz")
    X_train = features['train']
    X_test = features['test']

    # Load labels
    labels = np.load(f"{FEATURES_DIR}/labels.npz")
    y_train = labels['train']
    y_test = labels['test']

    # Load label encoder
    with open(f"{FEATURES_DIR}/label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)

    print(f"Features loaded: {X_train.shape}")
    print(f"Classes: {len(le.classes_)}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    has_features = True
except Exception as e:
    print(f"Features not found: {e}")
    print("Run feature_extraction.py first!")
    has_features = False

if not has_features:
    exit()

# Convert labels to categorical
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ============================================================
# MODEL 1: SIMPLE DNN
# ============================================================
print("\n" + "="*60)
print("MODEL 1: SIMPLE DEEP NEURAL NETWORK (DNN)")
print("="*60)


def build_dnn(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Build
model_dnn = build_dnn(X_train.shape[1], num_classes)
print("\nDNN Architecture:")
model_dnn.summary()

# Train
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("\nTraining DNN...")
history_dnn = model_dnn.fit(
    X_train, y_train_cat,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\n[DNN Evaluation]")
y_pred_dnn = model_dnn.predict(X_test)
y_pred_classes_dnn = np.argmax(y_pred_dnn, axis=1)

acc_dnn = accuracy_score(y_test, y_pred_classes_dnn)
print(f"Accuracy: {acc_dnn:.4f} ({acc_dnn*100:.2f}%)")

# Save
model_dnn.save(f"{OUTPUT_DIR}/dnn_model.keras")
print(f"Model saved: {OUTPUT_DIR}/dnn_model.keras")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n[Generating Plots]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy
axes[0,0].plot(history_dnn.history['accuracy'], label='Train')
axes[0,0].plot(history_dnn.history['val_accuracy'], label='Val')
axes[0,0].set_title('DNN Accuracy')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0,1].plot(history_dnn.history['loss'], label='Train')
axes[0,1].plot(history_dnn.history['val_loss'], label='Val')
axes[0,1].set_title('DNN Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes_dnn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1,0].set_title('Confusion Matrix')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('True')

# Plot 4: Classification Report as heatmap
report = classification_report(y_test, y_pred_classes_dnn, 
                               target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Exclude support/avg
sns.heatmap(report_df, annot=True, cmap='YlOrRd', ax=axes[1,1])
axes[1,1].set_title('Precision/Recall/F1 per Class')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dnn_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes_dnn, target_names=le.classes_))

# ============================================================
# SAVE METADATA
# ============================================================
metadata = {
    'model': 'Simple DNN',
    'accuracy': float(acc_dnn),
    'input_dim': int(X_train.shape[1]),
    'num_classes': int(num_classes),
    'classes': list(le.classes_),
    'training_epochs': len(history_dnn.history['loss']),
    'final_train_acc': float(history_dnn.history['accuracy'][-1]),
    'final_val_acc': float(history_dnn.history['val_accuracy'][-1])
}

with open(f"{OUTPUT_DIR}/model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Model: {OUTPUT_DIR}/dnn_model.keras")
print(f"Plots: {OUTPUT_DIR}/dnn_evaluation.png")
print(f"Metadata: {OUTPUT_DIR}/model_metadata.json")
print(f"\nFinal Accuracy: {acc_dnn:.4f} ({acc_dnn*100:.2f}%)")