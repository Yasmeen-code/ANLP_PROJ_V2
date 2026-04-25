import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

# Config
INPUT_PATH = "processed_data/cleaned.csv"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load
df = pd.read_csv(INPUT_PATH)
print(f"Loaded: {df.shape}")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['Category'])
print(f"Classes: {len(le.classes_)}")

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['cleaned'], y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

# ============================================================
# 1. TF-IDF (Essential)
# ============================================================
print("\n[1. TF-IDF]")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print(f"Shape: {X_train_tfidf.shape}")
print(f"Vocab: {len(tfidf.vocabulary_)} words")

# ============================================================
# 2. Word Embeddings (Optional but recommended)
# ============================================================
print("\n[2. Word Embeddings]")

try:
    import spacy
    nlp = spacy.load("en_core_web_md")

    def get_vectors(texts):
        return np.array([nlp(t).vector if nlp(t).has_vector else np.zeros(300) for t in texts])

    X_train_emb = get_vectors(X_train_text)
    X_test_emb = get_vectors(X_test_text)

    print(f"Shape: {X_train_emb.shape} (300 dims)")
    has_emb = True
except Exception as e:
    print(f"spaCy not available: {e}")
    has_emb = False

# ============================================================
# 3. Combine (TF-IDF + Embeddings)
# ============================================================
print("\n[3. Combine]")

if has_emb:
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_emb)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_emb)])
    print(f"Combined: {X_train.shape}")
else:
    X_train, X_test = X_train_tfidf, X_test_tfidf
    print(f"TF-IDF only: {X_train.shape}")

# ============================================================
# 4. Save
# ============================================================
print("\n[4. Save]")

np.savez(f"{OUTPUT_DIR}/features.npz", train=X_train, test=X_test)
np.savez(f"{OUTPUT_DIR}/labels.npz", train=y_train, test=y_test)

with open(f"{OUTPUT_DIR}/label_encoder.pkl", 'wb') as f:
    pickle.dump(le, f)
with open(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(tfidf, f)

with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
    json.dump({'classes': list(le.classes_), 'shape': str(X_train.shape)}, f)

print(f"Saved to {OUTPUT_DIR}/")
print("  - features.npz")
print("  - labels.npz")
print("  - label_encoder.pkl")
print("  - tfidf_vectorizer.pkl")
print("Done!")