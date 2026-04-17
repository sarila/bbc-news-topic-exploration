# BBC Articles Data Science Analysis

## Project Overview
This project analyzes BBC news articles using natural language processing and topic modeling techniques.

## Steps Performed
1. **Data Loading**: Load BBC articles dataset from Kaggle
2. **Data Cleaning**: Tokenization and removal of stopwords
3. **Zipf's Law Analysis**: Visualize word frequency distribution
4. **Topic Modeling**: Apply LDA to discover topics in articles

## Setup Instructions

### 1. Install Required Packages
```bash
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 2. Download the Dataset
You have two options:

#### Option A: Manual Download
1. Go to https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
2. Download the dataset
3. Extract and save the CSV file as `bbc-news-data.csv` in this directory

#### Option B: Using Kaggle API
```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d jacopoferretti/bbc-articles-dataset
unzip bbc-articles-dataset.zip
```

### 3. Run the Analysis
```bash
python bbc_analysis.py
```

## Output Files

After running the script, you'll get:

1. **zipfs_law_plot.png** - Visualization of Zipf's law
2. **lda_topics.csv** - Topics discovered by LDA with top words
3. **topic_distribution.png** - Distribution of documents across topics
4. **topic_word_heatmap.png** - Heatmap of topic-word relationships
5. **bbc_articles_processed.csv** - Processed dataset with topic assignments

You can change the text column you are using to either:
1. 'text' column for original news text
2. 'text_rank_summary' column for a summary of news using text rank
3. 'lsa_summary' column for a summary of news using lsa

## Interpreting Results

See **TOPIC_INTERPRETATION.md** for a comprehensive guide on:
- Understanding LDA topics
- Interpreting Zipf's law plots
- Making sense of word frequencies
- Analyzing topic distributions

## Project Structure
```
.
├── bbc_analysis.py              # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── TOPIC_INTERPRETATION.md      # Topic interpretation guide
└── bbc-news-data.csv           # Dataset (you need to download)
```

## Technologies Used
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Visualization
- **spacy**: Natural language processing
- **scikit-learn**: Machine learning (LDA)

## Common Issues

### "Dataset file not found"
- Make sure you've downloaded the dataset
- Check the filename matches `bbc-news-data.csv`
- Ensure it's in the same directory as the script

### spaCy model not found
- Make sure you've downloaded the spaCy language model
- Run: `python -m spacy download en_core_web_sm`
- The script will check for the model at startup

## Author
Sarila Ngakhusi

## Dataset Credit
Dataset from: https://www.kaggle.com/datasets/jacopoferretti/bbc-articles-dataset
