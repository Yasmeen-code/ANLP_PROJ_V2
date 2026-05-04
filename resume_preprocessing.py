import os, re, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ===== Config =====
CSV_PATH = "resume_dataset.csv"
OUTPUT_DIR = "processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Load =====
df = pd.read_csv(CSV_PATH)
text_col = 'Resume_str' if 'Resume_str' in df.columns else df.select_dtypes(include='object').columns[0]
print(f"Loaded: {df.shape}, Text col: {text_col}, Classes: {df['Category'].nunique()}")

# ===== Preprocessing =====
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {'resume','cv','email','phone','contact','name','date','www','com','http','https','objective','summary','references','available','present','current'}

def clean(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+\.\S+|[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,4}[-\s\.]?[0-9]{1,9}|\b\d+\b|[^a-zA-Z\s]', ' ', text)
    tokens = [lemmatizer.lemmatize(t) for t in word_tokenize(text) if t not in stop_words and len(t) > 2 and t.isalpha()]
    return ' '.join(tokens)

df['cleaned'] = df[text_col].astype(str).apply(clean)
df = df[df['cleaned'].str.strip() != ''].reset_index(drop=True)

# ===== Stats =====
raw_words = df[text_col].astype(str).apply(lambda x: len(x.split()))
clean_words = df['cleaned'].apply(lambda x: len(x.split()))
print(f"Avg words: {raw_words.mean():.0f} -> {clean_words.mean():.0f}")
print(f"Vocab: {len(set(' '.join(df[text_col].astype(str)).split()))} -> {len(set(' '.join(df['cleaned']).split()))}")

# ===== Plots =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Category'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue', title='Class Distribution')
df.groupby('Category')['cleaned'].apply(lambda x: x.str.split().str.len().mean()).sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='coral', title='Avg Words/Category')
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/plots.png"); plt.show()

# ===== Save =====
df[['ID','Category',text_col,'cleaned']].to_csv(f"{OUTPUT_DIR}/cleaned.csv", index=False)
json.dump({'samples':len(df),'classes':df['Category'].nunique(),'avg_raw':float(raw_words.mean()),'avg_clean':float(clean_words.mean())}, open(f"{OUTPUT_DIR}/stats.json",'w'))
print("Done!")