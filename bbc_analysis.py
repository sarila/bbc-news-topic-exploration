"""
BBC Articles Analysis
This script performs text analysis on BBC articles dataset including:
- Data loading from Kaggle
- Text cleaning and tokenization
- Zipf's law visualization
- Topic modeling with LDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import re

# For text processing
import spacy

# For topic modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# For topic coherence
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

# Global variable for spaCy model
nlp = None

def load_spacy_model():
    """Load spaCy model"""
    global nlp
    if nlp is None:
        print("Loading spaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            print(" spaCy model loaded successfully!")
        except OSError:
            print("ERROR: spaCy model 'en_core_web_sm' not found!")
            print("Please install it using: python -m spacy download en_core_web_sm")
            exit()
    return nlp

# Note: You need to download the dataset from Kaggle first
# Method 1: Download manually from https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
# Method 2: Use Kaggle API (requires kaggle.json credentials)
# 
# To use Kaggle API:
# 1. Install kaggle: pip install kaggle
# 2. Get your API token from kaggle.com/account
# 3. Place kaggle.json in ~/.kaggle/
# 4. Run: kaggle datasets download -d jacopoferretti/bbc-articles-dataset

# For now, we'll assume you've downloaded the CSV file
try:
    # Try to load the dataset (update the path to where you saved it)
    df = pd.read_csv('bbc-news-data.csv')
    print(f"Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
except FileNotFoundError:
    print("ERROR: Dataset file not found!")
    print("Please download the dataset from:")
    print("https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset")
    print("and save it as 'bbc-news-data.csv' in the same directory.")
    exit()

# =============================================================================
# STEP 2: Data Cleaning with Tokenization and Removing Common Words
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Data Cleaning and Tokenization")
print("="*70)

def clean_text(text):
    """
    Clean and preprocess text data using spaCy
    - Convert to lowercase
    - Remove punctuation
    - Remove numbers
    - Tokenize
    - Remove stopwords
    """
    if pd.isna(text):
        return []
    
    # Process text with spaCy
    doc = nlp(text.lower())
    
    # Extract tokens: remove stopwords, punctuation, numbers, and short words
    cleaned_tokens = [
        token.text for token in doc 
        if not token.is_stop           # Remove stopwords
        and not token.is_punct         # Remove punctuation
        and not token.like_num         # Remove numbers
        and len(token.text) > 2        # Remove short words
        and token.text.isalpha()       # Keep only alphabetic tokens
    ]
    
    return cleaned_tokens

def calculate_coherence_scores(documents, topic_range=range(30, 49)):
    # Prepare documents as list of token lists for gensim
    tokenized_docs = [doc.split() for doc in documents if doc.strip()]
    
    # Create gensim dictionary and corpus
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    
    coherence_scores = []
    models = []
    
    print(f"\nCalculating coherence scores for {len(topic_range)} different topic counts...")
    print("This may take a few minutes...\n")
    
    for num_topics in topic_range:
        print(f"Testing {num_topics} topics...", end=" ")
        
        # Train gensim LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # Disable multiprocessing to avoid spawn issues
        )
        
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(lda_model)
        
        print(f"Coherence: {coherence_score:.4f}")
    
    # Find optimal number of topics
    optimal_idx = np.argmax(coherence_scores)
    optimal_topics = list(topic_range)[optimal_idx]
    optimal_score = coherence_scores[optimal_idx]
    
    print(f"\n Optimal number of topics: {optimal_topics} (coherence: {optimal_score:.4f})")
    
    return {
        'topic_numbers': list(topic_range),
        'coherence_scores': coherence_scores,
        'optimal_topics': optimal_topics,
        'optimal_score': optimal_score,
        'best_model': models[optimal_idx]
    }

def plot_coherence_scores(topic_numbers, coherence_scores, optimal_topics):
    """
    Visualize coherence scores across different numbers of topics.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(topic_numbers, coherence_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_topics, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {optimal_topics} topics')
    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel('Coherence Score (C_v)', fontsize=12)
    plt.title('Topic Coherence Scores: Finding Optimal Number of Topics', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(topic_numbers)
    plt.tight_layout()
    plt.savefig('coherence_scores_main_text.png', dpi=300, bbox_inches='tight')
    print(" Coherence scores plot saved as 'coherence_scores_main_text.png'")
    plt.show()

def main():
    """Main execution function"""
    # Load spaCy model
    global nlp
    nlp = load_spacy_model()
    
    # STEP 1: Load the data from Kaggle
    print("STEP 1: Loading Data")
    try:
        # Try to load the dataset (update the path to where you saved it)
        df = pd.read_csv('bbc-news-data.csv')
        print(f"Dataset loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
    except FileNotFoundError:
        print("ERROR: Dataset file not found!")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset")
        print("and save it as 'bbc-news-data.csv' in the same directory.")
        exit()
    
    # STEP 2: Data Cleaning with Tokenization and Removing Common Words
    print("STEP 2: Data Cleaning and Tokenization")
    
    # Text column changes depending on the content.
    text_column = 'text'
    # text_column = 'text_rank_summary'
    # text_column = 'lsa_summary'
    
    print(f"\nUsing column: '{text_column}'")
    
    # Apply cleaning to all articles
    print("Cleaning and tokenizing articles...")
    df['cleaned_tokens'] = df[text_column].apply(clean_text)
    
    # Combine all tokens for analysis
    all_tokens = []
    for tokens in df['cleaned_tokens']:
        all_tokens.extend(tokens)
    
    print(f" Cleaning complete!")
    print(f"  - Total tokens after cleaning: {len(all_tokens)}")
    print(f"  - Unique words: {len(set(all_tokens))}")
    
    # STEP 3: Implement Zipf's Law (Rank vs Frequency)
    print("STEP 3: Zipf's Law Analysis")    
    word_freq = Counter(all_tokens)
    words_sorted = word_freq.most_common()
    
    # Prepare data for plotting
    ranks = list(range(1, len(words_sorted) + 1))
    frequencies = [freq for word, freq in words_sorted]
    
    print(f" Word frequency analysis complete!")
    print(f"\nTop 10 most frequent words:")
    for i, (word, freq) in enumerate(words_sorted[:10], 1):
        print(f"  {i}. '{word}': {freq} occurrences")
    
    # Create Zipf's Law plot
    plt.figure(figsize=(14, 5))
    
    
    # Plot 1: Linear scale
    plt.subplot(1, 2, 1)
    plt.plot(ranks[:100], frequencies[:100], 'b-', linewidth=2)
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title("Zipf's Law: Rank vs Frequency (Linear Scale)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale (Zipf's law appears as straight line)
    plt.subplot(1, 2, 2)
    plt.loglog(ranks, frequencies, 'r-', linewidth=2)
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title("Zipf's Law: Rank vs Frequency (Log-Log Scale)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zipfs_law_plot_main_text.png', dpi=300, bbox_inches='tight')
    print("\n Zipf's Law plots saved as 'zipfs_law_plot_main_text.png'")
    plt.show()
    
    # =============================================================================
    # STEP 3.5: Find Optimal Number of Topics using Coherence Score
    # =============================================================================
    print("\n" + "="*70)
    print("STEP 3.5: Finding Optimal Number of Topics")
    print("="*70)
    
    # Prepare documents (join tokens back into strings)
    documents = [' '.join(tokens) for tokens in df['cleaned_tokens']]
    
    # Remove empty documents
    documents = [doc for doc in documents if doc.strip()]
    
    print(f"Processing {len(documents)} documents...")
    
    # Calculate coherence scores for different numbers of topics
    coherence_results = calculate_coherence_scores(
        documents=documents,
        topic_range=range(30, 49)  # Test 30-49 topics
    )
    
    # Plot results
    plot_coherence_scores(
        topic_numbers=coherence_results['topic_numbers'],
        coherence_scores=coherence_results['coherence_scores'],
        optimal_topics=coherence_results['optimal_topics']
    )
    
    # Use optimal number of topics for LDA
    n_topics = coherence_results['optimal_topics']
    print(f"\n Using {n_topics} topics for LDA model (based on coherence score)")
    
    print("STEP 4: Topic Modeling with LDA")
    print(f"Processing {len(documents)} documents for topic modeling...")
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,  # Ignore words that appear in >95% of documents
        min_df=2,     # Ignore words that appear in <2 documents
        max_features=1000  # Limit to top 1000 features
    )
    
    doc_term_matrix = vectorizer.fit_transform(documents)
    print(f" Document-term matrix created: {doc_term_matrix.shape}")
    
    # Train LDA model (n_topics already determined by coherence score)
    print(f"\nTraining LDA model with {n_topics} topics...")
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=30,
        learning_method='online',
        n_jobs=-1
    )
    
    lda_output = lda_model.fit_transform(doc_term_matrix)
    print(" LDA model trained successfully!")
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Display topics
    print("\n" + "="*70)
    print("DISCOVERED TOPICS")
    print("="*70)
    
    n_top_words = 10
    topics_data = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        
        topics_data.append({
            'topic_num': topic_idx + 1,
            'words': top_words,
            'weights': top_weights
        })
        
        print(f"\nTopic {topic_idx + 1}:")
        print(f"  Top words: {', '.join(top_words)}")
    
    # Save topics to CSV for documentation
    topics_df = pd.DataFrame([
        {
            'Topic': data['topic_num'],
            'Top_Words': ', '.join(data['words']),
            'Word_1': data['words'][0],
            'Word_2': data['words'][1],
            'Word_3': data['words'][2],
            'Word_4': data['words'][3],
            'Word_5': data['words'][4]
        }
        for data in topics_data
    ])
    topics_df.to_csv('lda_topics_main_text.csv', index=False)
    print("\n Topics saved to 'lda_topics_main_text.csv'")

    # Assign dominant topic to each document
    dominant_topics = np.argmax(lda_output, axis=1)
    df['dominant_topic'] = dominant_topics
    
    # Topic distribution visualization
    plt.figure(figsize=(10, 6))
    # Count documents per topic and ensure an entry for every topic index
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    # Reindex so that we have a count for every topic index (fill missing with 0)
    topic_counts = topic_counts.reindex(range(n_topics), fill_value=0)
    x_vals = list(range(n_topics))
    plt.bar(x_vals, topic_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Topic Number', fontsize=12)
    plt.ylabel('Number of Documents', fontsize=12)
    plt.title('Distribution of Documents Across Topics', fontsize=14, fontweight='bold')
    plt.xticks(x_vals, [f'Topic {i+1}' for i in x_vals], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('topic_distribution_main_text.png', dpi=300, bbox_inches='tight')
    print(" Topic distribution plot saved as 'topic_distribution_main_text.png'")
    plt.show()
    
    # Heatmap of topic-word relationships
    plt.figure(figsize=(24, 18))
    top_n_words_heatmap = 15
    top_word_indices = []
    for topic in lda_model.components_:
        top_word_indices.extend(topic.argsort()[-top_n_words_heatmap:])
    top_word_indices = list(set(top_word_indices))
    
    topic_word_matrix = lda_model.components_[:, top_word_indices]
    words_for_heatmap = [feature_names[i] for i in top_word_indices]
    
    sns.heatmap(topic_word_matrix, 
                xticklabels=words_for_heatmap,
                yticklabels=[f'Topic {i+1}' for i in range(n_topics)],
                cmap='YlOrRd',
                cbar_kws={'label': 'Word Importance'})
    plt.title('Topic-Word Relationship Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Topics', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig('topic_word_heatmap_main_text.png', dpi=300, bbox_inches='tight')
    print(" Topic-word heatmap saved as 'topic_word_heatmap_main_text.png'")
    plt.show()
    
    # Save processed data
    df.to_csv('bbc_articles_processed_main_text.csv', index=False)
    print("\n Processed data saved as 'bbc_articles_processed_main_text.csv'")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. zipfs_law_plot_main_text.png - Visualization of Zipf's law")
    print("  2. coherence_scores_main_text.png - Coherence scores for different topic counts")
    print("  3. lda_topics_main_text.csv - LDA topics with top words")
    print("  4. topic_distribution_main_text.png - Distribution of documents across topics")
    print("  5. topic_word_heatmap_main_text.png - Heatmap showing topic-word relationships")
    print("  6. bbc_articles_processed_main_text.csv - Processed dataset with topic assignments")

if __name__ == '__main__':
    main()