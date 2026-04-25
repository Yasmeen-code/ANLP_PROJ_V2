import os, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Config
CSV_PATH = "resume_dataset.csv"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Text column auto-detect
text_col = 'Resume_str' if 'Resume_str' in df.columns else [c for c in df.columns if df[c].dtype == 'object'][0]
print(f"Text column: {text_col}")

# Class distribution
print(f"\nClasses: {df['Category'].nunique()}")
print(df['Category'].value_counts())

# Plot raw distribution
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Class Distribution (Raw)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png")
plt.show()

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_sw = {'resume', 'cv', 'curriculum', 'vitae', 'page', 'email', 'phone', 'address', 
             'contact', 'name', 'date', 'www', 'com', 'http', 'https', 'objective', 
             'summary', 'references', 'available', 'present', 'current'}
stop_words.update(custom_sw)

def clean(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2 and t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Process
print("\nProcessing...")
df['cleaned'] = df[text_col].astype(str).apply(clean)

# Remove empty
df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)

# Stats
raw_words = df[text_col].astype(str).apply(lambda x: len(x.split()))
clean_words = df['cleaned'].apply(lambda x: len(x.split()))

print(f"\n[BEFORE -> AFTER]")
print(f"Avg words: {raw_words.mean():.0f} -> {clean_words.mean():.0f} ({((clean_words.mean()-raw_words.mean())/raw_words.mean()*100):+.1f}%)")
print(f"Vocab: {len(set(' '.join(df[text_col].astype(str)).split()))} -> {len(set(' '.join(df['cleaned']).split()))}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(raw_words, bins=50, alpha=0.7, label='Before')
axes[0].hist(clean_words, bins=50, alpha=0.7, label='After')
axes[0].set_title('Word Count')
axes[0].legend()
df.groupby('Category')['cleaned'].apply(lambda x: x.str.split().str.len().mean()).sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Avg Words/Category (After)')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_comparison.png")
plt.show()

# Save
cols = ['ID', 'Category', text_col, 'cleaned']
cols = [c for c in cols if c in df.columns]
df[cols].to_csv(f"{OUTPUT_DIR}/cleaned.csv", index=False)

with open(f"{OUTPUT_DIR}/stats.json", 'w') as f:
    json.dump({
        'samples': len(df),
        'classes': df['Category'].nunique(),
        'avg_raw_words': float(raw_words.mean()),
        'avg_clean_words': float(clean_words.mean()),
        'vocab_raw': len(set(' '.join(df[text_col].astype(str)).split())),
        'vocab_clean': len(set(' '.join(df['cleaned']).split()))
    }, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
print("Done!")