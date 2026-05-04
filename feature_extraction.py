import os, json, pickle
import numpy as np, pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "processed_data/cleaned.csv"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
le = LabelEncoder()
y = le.fit_transform(df['Category'])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['cleaned'], y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

# Save sparse properly
save_npz(f"{OUTPUT_DIR}/train_features.npz", X_train)
save_npz(f"{OUTPUT_DIR}/test_features.npz", X_test)
np.savez(f"{OUTPUT_DIR}/labels.npz", train=y_train, test=y_test)
pickle.dump(le, open(f"{OUTPUT_DIR}/label_encoder.pkl", 'wb'))
pickle.dump(tfidf, open(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl", 'wb'))

print(f"Done! Train: {X_train.shape}, Test: {X_test.shape}")