# BBC Articles Analysis (Colab-friendly)
# This file mirrors `bbc_analysis.py` but is organized into small, copy-paste friendly
# sections ("cells") for Google Colab. Each section is labeled "COLAB CELL" and can be
# pasted into a separate Colab cell to run step-by-step.

# === COLAB CELL 1: Install required packages (run once) ===
import subprocess
import sys

def pip_install(packages):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)

# Recommended packages. In Colab this will install if missing; locally it is safe to run too.
try:
    pip_install([
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'spacy',
        'scikit-learn', 'kaggle', 'python-dotenv'
    ])
except Exception as e:
    print('Package install encountered an issue (you can re-run this cell or install manually):', e)

# Download spaCy model if not present
try:
    import spacy
    spacy.load('en_core_web_sm')
except Exception:
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    except Exception as e:
        print('Failed to download spaCy model automatically. Run `python -m spacy download en_core_web_sm` manually.', e)

print('COLAB CELL 1 complete: packages installed / spaCy model ensured.')

# === COLAB CELL 2: Imports and setup ===
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import spacy

sns.set(style='whitegrid')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    print('spaCy model loaded.')
except Exception as e:
    print('spaCy model not loaded:', e)

# === COLAB CELL 3: Load dataset (options for Colab) ===
# Option A (recommended in Colab): Upload the CSV directly to the notebook runtime
#   from google.colab import files
#   uploaded = files.upload()
#   df = pd.read_csv(next(iter(uploaded.keys())))

# Option B: Mount Google Drive and read file from Drive
#   from google.colab import drive
#   drive.mount('/content/drive')
#   df = pd.read_csv('/content/drive/MyDrive/path/to/bbc-news-data.csv')

# Option C: If running locally, place 'bbc-news-data.csv' in the same folder and it will be loaded.

# The code below tries to detect Colab and mount/upload; otherwise it falls back to local file.

def load_bbc_csv(default_filename='bbc-news-data.csv'):
    # Try Colab upload
    try:
        import google.colab
        from google.colab import files
        print('Detected Colab. Please upload the dataset file when prompted.')
        uploaded = files.upload()
        if uploaded:
            fname = next(iter(uploaded.keys()))
            return pd.read_csv(fname)
    except Exception:
        pass

    # Try mounting Drive (if user prefers)
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        # If user placed file in Drive root or specify path, adjust below
        drive_path = os.path.join('/content/drive/MyDrive', default_filename)
        if os.path.exists(drive_path):
            return pd.read_csv(drive_path)
    except Exception:
        pass

    # Fallback: local file
    if os.path.exists(default_filename):
        return pd.read_csv(default_filename)

    raise FileNotFoundError(f"Dataset not found. Place '{default_filename}' in notebook, mount Drive, or upload in Colab.")

# Load dataset
try:
    df = load_bbc_csv()
    print('Dataset loaded. Shape:', df.shape)
    print('Columns:', df.columns.tolist())
except Exception as e:
    print('Failed to load dataset:', e)
    # Stop here in Colab to allow user to upload or mount Drive
    raise

# === COLAB CELL 4: Data cleaning function and tokenization ===
print('\nSTEP: Data Cleaning and Tokenization')

# Clean text using spaCy. Returns list of cleaned tokens.
def clean_text(text):
    if pd.isna(text):
        return []
    # Lowercase and process
    doc = nlp(str(text).lower())
    cleaned_tokens = [
        token.text for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num
        and len(token.text) > 2 and token.text.isalpha()
    ]
    return cleaned_tokens

# Identify text column
text_column = None
for col in ['content', 'text', 'article', 'description']:
    if col in df.columns:
        text_column = col
        break

if text_column is None:
    raise ValueError('Could not detect text column. Available columns: ' + str(df.columns.tolist()))

print('Using text column:', text_column)

# Apply cleaning
df['cleaned_tokens'] = df[text_column].apply(clean_text)

# Combine tokens
all_tokens = [token for tokens in df['cleaned_tokens'] for token in tokens]
print('Total tokens:', len(all_tokens), 'Unique words:', len(set(all_tokens)))

# === COLAB CELL 5: Zipf's Law visualization ===
print('\nSTEP: Zipf\'s Law Analysis')
word_freq = Counter(all_tokens)
words_sorted = word_freq.most_common()
ranks = list(range(1, len(words_sorted) + 1))
frequencies = [freq for word, freq in words_sorted]

print('\nTop 10 words:')
for i, (word, freq) in enumerate(words_sorted[:10], 1):
    print(f" {i}. '{word}': {freq}")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(ranks[:100], frequencies[:100], 'b-', linewidth=2)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf's Law: Rank vs Frequency (Linear)")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.loglog(ranks, frequencies, 'r-', linewidth=2)
plt.xlabel('Rank (log)')
plt.ylabel('Frequency (log)')
plt.title("Zipf's Law: Rank vs Frequency (Log-Log)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('zipfs_law_plot.png', dpi=300, bbox_inches='tight')
print("Saved 'zipfs_law_plot.png'")
plt.show()

# === COLAB CELL 6: Topic Modeling with LDA ===
print('\nSTEP: Topic Modeling with LDA')
# Prepare documents
documents = [' '.join(toks) for toks in df['cleaned_tokens']]
documents = [doc for doc in documents if doc.strip()]
print('Processing', len(documents), 'documents')

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
doc_term_matrix = vectorizer.fit_transform(documents)
print('Document-term matrix shape:', doc_term_matrix.shape)

n_topics = 10
lda_model = LatentDirichletAllocation(
    n_components=n_topics, random_state=42, max_iter=20, learning_method='online'
)
lda_output = lda_model.fit_transform(doc_term_matrix)
print('LDA trained')

feature_names = vectorizer.get_feature_names_out()

n_top_words = 10
topics_data = []
for topic_idx, topic in enumerate(lda_model.components_):
    top_idx = topic.argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_idx]
    topics_data.append({'topic_num': topic_idx + 1, 'words': top_words})
    print(f"\nTopic {topic_idx + 1}: {', '.join(top_words)}")

# Save topics
topics_df = pd.DataFrame([
    {'Topic': d['topic_num'], 'Top_Words': ', '.join(d['words'])} for d in topics_data
])
topics_df.to_csv('lda_topics.csv', index=False)
print("Saved 'lda_topics.csv'")

# Assign dominant topic to each original row (only for rows that had non-empty documents)
# We need to map back to original df indices that correspond to 'documents'.
non_empty_mask = df['cleaned_tokens'].apply(lambda x: len(x) > 0)
indices_non_empty = df[non_empty_mask].index

# Create an array of -1 and fill values for non-empty documents
dominant_topics = np.full(shape=(len(df),), fill_value=-1, dtype=int)
if len(indices_non_empty) == lda_output.shape[0]:
    dominant_topics[indices_non_empty] = np.argmax(lda_output, axis=1)
else:
    # Safer mapping if any mismatch (shouldn't happen)
    dominant_topics[indices_non_empty[:lda_output.shape[0]]] = np.argmax(lda_output, axis=1)

df['dominant_topic'] = dominant_topics

# Topic distribution plot
plt.figure(figsize=(10, 6))
topic_counts = pd.Series([t for t in dominant_topics if t >= 0]).value_counts().sort_index()
plt.bar(range(n_topics), topic_counts.reindex(range(n_topics), fill_value=0), color='steelblue', edgecolor='black')
plt.xlabel('Topic Number')
plt.ylabel('Number of Documents')
plt.title('Distribution of Documents Across Topics')
plt.xticks(range(n_topics), [f'Topic {i+1}' for i in range(n_topics)])
plt.tight_layout()
plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
print("Saved 'topic_distribution.png'")
plt.show()

# Heatmap of topic-word relationships
plt.figure(figsize=(12, 8))
top_n_words_heatmap = 15
top_word_indices = []
for topic in lda_model.components_:
    top_word_indices.extend(topic.argsort()[-top_n_words_heatmap:])
top_word_indices = list(dict.fromkeys(top_word_indices))  # preserve order and dedupe

topic_word_matrix = lda_model.components_[:, top_word_indices]
words_for_heatmap = [feature_names[i] for i in top_word_indices]

sns.heatmap(topic_word_matrix, xticklabels=words_for_heatmap,
            yticklabels=[f'Topic {i+1}' for i in range(n_topics)], cmap='YlOrRd',
            cbar_kws={'label': 'Word Importance'})
plt.title('Topic-Word Relationship Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('topic_word_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved 'topic_word_heatmap.png'")
plt.show()

# === COLAB CELL 7: Save processed data and wrap up ===
df.to_csv('bbc_articles_processed.csv', index=False)
print("Saved 'bbc_articles_processed.csv'")

print('\nANALYSIS COMPLETE. Generated files:')
print(' - zipfs_law_plot.png')
print(' - lda_topics.csv')
print(' - topic_distribution.png')
print(' - topic_word_heatmap.png')
print(' - bbc_articles_processed.csv')
print("\nTip: Paste each COLAB CELL section into a separate Colab cell and run in order.")
