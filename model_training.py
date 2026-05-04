import os, pickle, json
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

FEATURES_DIR = "features"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading features...")
X_train = load_npz(f"{FEATURES_DIR}/train_features.npz")
X_test = load_npz(f"{FEATURES_DIR}/test_features.npz")

labels = np.load(f"{FEATURES_DIR}/labels.npz")
y_train, y_test = labels['train'], labels['test']

with open(f"{FEATURES_DIR}/label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {len(le.classes_)}")

results = {}

# A. Logistic Regression
print("\n[A] Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
results['Logistic Regression'] = {'acc': lr_acc, 'f1': lr_f1, 'model': lr}
print(f"    Acc: {lr_acc:.4f} | F1: {lr_f1:.4f}")

# B. Linear SVM
print("\n[B] Linear SVM...")
svm = LinearSVC(max_iter=2000, C=0.5, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
results['Linear SVM'] = {'acc': svm_acc, 'f1': svm_f1, 'model': svm}
print(f"    Acc: {svm_acc:.4f} | F1: {svm_f1:.4f}")

# C. DNN
print("\n[C] DNN (MLP)...")
dnn = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu', solver='adam', alpha=1e-4,
    batch_size=64, learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=100, random_state=42,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=10
)
dnn.fit(X_train, y_train)
dnn_pred = dnn.predict(X_test)
dnn_acc = accuracy_score(y_test, dnn_pred)
dnn_f1 = f1_score(y_test, dnn_pred, average='weighted')
results['DNN (MLP)'] = {'acc': dnn_acc, 'f1': dnn_f1, 'model': dnn}
print(f"    Acc: {dnn_acc:.4f} | F1: {dnn_f1:.4f}")

# Best
best = max(results, key=lambda x: results[x]['acc'])
print(f"\n{'='*50}\nBEST: {best} | Acc: {results[best]['acc']:.4f}\n{'='*50}")

print(f"\nReport ({best}):\n{classification_report(y_test, results[best]['model'].predict(X_test), target_names=le.classes_)}")

# Save
for name, data in results.items():
    with open(f"{OUTPUT_DIR}/{name.replace(' ', '_').lower()}.pkl", 'wb') as f:
        pickle.dump(data['model'], f)

with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
    json.dump({k: {'acc': v['acc'], 'f1': v['f1']} for k, v in results.items()}, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")