## @file finbert_sentiment.py
# @brief Stage 1: FinBERT Sentiment Extraction
#
# Applies FinBERT sentiment analysis to all REIT Glassdoor reviews.
#
# @details
# Input:  data/raw/all_reit_reviews_merged.csv
# Output: data/processed/finbert_sentiment_scores.csv
#
# FinBERT: Financial sentiment model fine-tuned on financial news
# - Trained on Financial PhraseBank dataset
# - Classifies text as positive, negative, or neutral
# - Provides probability scores for each class
#
# @author Konain Niaz (kn4792@rit.edu)
# @date 2025-12-01

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os
import json
from datetime import datetime

print("=" * 80)
print("STAGE 1: FINBERT SENTIMENT EXTRACTION")
print("=" * 80)
print()

## @var INPUT_FILE
# Path to the input CSV file containing raw Glassdoor reviews
INPUT_FILE = "data/raw/all_reit_reviews_merged.csv"

## @var OUTPUT_DIR
# Directory where processed output files will be saved
OUTPUT_DIR = "data/processed"

## @var OUTPUT_FILE
# Path to the output CSV file with sentiment scores
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finbert_sentiment_scores.csv")

## @var BATCH_SIZE
# Number of reviews to process in each batch for memory efficiency
BATCH_SIZE = 32

## @var MAX_LENGTH
# Maximum sequence length for FinBERT tokenizer (model limit is 512)
MAX_LENGTH = 512

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# Load the raw Glassdoor reviews from CSV file
# ============================================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} reviews")
print(f"Columns: {df.columns.tolist()}")
print()

# ============================================================================
# STEP 2: PREPARE TEXT
# Combine pros and cons into single text for sentiment analysis
# ============================================================================
print("Preparing text for sentiment analysis...")


## Combine pros and cons into single text for sentiment analysis.
#
# This function takes a row from the DataFrame and combines the 'pros' and 
# 'cons' columns into a single labeled text string suitable for FinBERT.
#
# @param row A pandas Series representing a single review row with 'pros' and 'cons' columns
# @return A formatted string combining pros and cons with semantic labels, or empty string if both are missing
def prepare_text(row):
    pros = str(row['pros']) if pd.notna(row['pros']) else ""
    cons = str(row['cons']) if pd.notna(row['cons']) else ""
    
    # Label the sections clearly for FinBERT to understand context
    if pros and cons:
        return f"Positive aspects: {pros} Negative aspects: {cons}"
    elif pros:
        return f"Positive aspects: {pros}"
    elif cons:
        return f"Negative aspects: {cons}"
    else:
        return ""


df['combined_text'] = df.apply(prepare_text, axis=1)

# Remove empty reviews - these cannot be analyzed
df_valid = df[df['combined_text'].str.len() > 0].copy()
print(f"Prepared {len(df_valid):,} reviews with valid text")
print(f"Excluded {len(df) - len(df_valid):,} empty reviews")
print()

# ============================================================================
# STEP 3: LOAD FINBERT MODEL
# Initialize the pre-trained FinBERT model and tokenizer
# ============================================================================
print("Loading FinBERT model...")
print("Model: ProsusAI/finbert")
print()

# Load tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Move to GPU if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout)

print(f"Model loaded on: {device}")
print()

# ============================================================================
# STEP 4: RUN SENTIMENT ANALYSIS
# Process all reviews through FinBERT in batches
# ============================================================================
print(f"Running sentiment analysis on {len(df_valid):,} reviews...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max sequence length: {MAX_LENGTH}")
print()

## @var id2label
# Mapping from FinBERT output indices to sentiment labels
id2label = {0: "positive", 1: "negative", 2: "neutral"}

# Store results for all reviews
results = []

# Convert text column to list for batch processing
texts = df_valid['combined_text'].tolist()
n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

# Process batches with no gradient computation (inference mode)
with torch.no_grad():
    for i in tqdm(range(n_batches), desc="Processing batches"):
        # Calculate batch indices
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # Tokenize batch with padding and truncation
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass through model
        outputs = model(**inputs)
        # Convert logits to probabilities using softmax
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract results for each review in batch
        for j, probs in enumerate(predictions):
            probs_cpu = probs.cpu().numpy()
            
            # Determine predicted sentiment label
            pred_idx = np.argmax(probs_cpu)
            sentiment_label = id2label[pred_idx]
            
            # Calculate sentiment score: positive - negative (range: -1 to 1)
            sentiment_score = probs_cpu[0] - probs_cpu[1]
            
            results.append({
                'positive_prob': float(probs_cpu[0]),
                'negative_prob': float(probs_cpu[1]),
                'neutral_prob': float(probs_cpu[2]),
                'sentiment_label': sentiment_label,
                'sentiment_score': float(sentiment_score)
            })

print("Sentiment analysis complete!")
print()

# ============================================================================
# STEP 5: COMBINE RESULTS WITH ORIGINAL DATA
# Merge sentiment scores back to the original DataFrame
# ============================================================================
print("Combining results with original data...")

# Create results dataframe from list of dictionaries
results_df = pd.DataFrame(results)

# Combine with valid reviews (reset index to align)
df_valid = df_valid.reset_index(drop=True)
df_valid = pd.concat([df_valid, results_df], axis=1)

# Merge back to include excluded reviews (they will have NaN sentiment)
df_final = df.merge(
    df_valid[['title', 'date', 'company', 'ticker', 
              'positive_prob', 'negative_prob', 'neutral_prob',
              'sentiment_label', 'sentiment_score']],
    on=['title', 'date', 'company', 'ticker'],
    how='left'
)

print(f"Final dataset: {len(df_final):,} reviews")
print()

# ============================================================================
# STEP 6: CALCULATE SUMMARY STATISTICS
# Compute and display sentiment distribution metrics
# ============================================================================
print("Summary Statistics:")
print()

print("Sentiment Distribution:")
sentiment_counts = df_final['sentiment_label'].value_counts()
for label, count in sentiment_counts.items():
    pct = count / len(df_final[df_final['sentiment_label'].notna()]) * 100
    print(f"  {label:>8}: {count:>6,} ({pct:>5.1f}%)")

print()
print("Sentiment Score Statistics:")
print(f"  Mean:   {df_final['sentiment_score'].mean():>7.4f}")
print(f"  Median: {df_final['sentiment_score'].median():>7.4f}")
print(f"  Std:    {df_final['sentiment_score'].std():>7.4f}")
print(f"  Min:    {df_final['sentiment_score'].min():>7.4f}")
print(f"  Max:    {df_final['sentiment_score'].max():>7.4f}")
print()

# ============================================================================
# STEP 7: SAVE RESULTS
# Export processed data and summary statistics to files
# ============================================================================
print("Saving results...")

# Save full dataset with sentiment scores
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")

# Save summary statistics as JSON for programmatic access
stats = {
    'n_reviews': len(df_final),
    'n_with_sentiment': int(df_final['sentiment_label'].notna().sum()),
    'n_excluded': int(df_final['sentiment_label'].isna().sum()),
    'sentiment_distribution': sentiment_counts.to_dict(),
    'sentiment_score_stats': {
        'mean': float(df_final['sentiment_score'].mean()),
        'median': float(df_final['sentiment_score'].median()),
        'std': float(df_final['sentiment_score'].std()),
        'min': float(df_final['sentiment_score'].min()),
        'max': float(df_final['sentiment_score'].max())
    },
    'timestamp': datetime.now().isoformat()
}

stats_file = os.path.join(OUTPUT_DIR, "finbert_summary_stats.json")
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Saved: {stats_file}")
print()

# ============================================================================
# STEP 8: DISPLAY SAMPLE RESULTS
# Show example reviews for each sentiment category
# ============================================================================
print("Sample Results:")
print()

# Show a few examples of each sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    sample = df_final[df_final['sentiment_label'] == sentiment].sample(1, random_state=42)
    
    print(f"{sentiment.upper()} Example:")
    print(f"  Company: {sample.iloc[0]['company']}")
    print(f"  Rating: {sample.iloc[0]['rating']}")
    print(f"  Sentiment Score: {sample.iloc[0]['sentiment_score']:.3f}")
    print(f"  Pros: {str(sample.iloc[0]['pros'])[:100]}...")
    print(f"  Cons: {str(sample.iloc[0]['cons'])[:100]}...")
    print()

print("=" * 80)
print("STAGE 1 COMPLETE")
print("=" * 80)
print()
print("Output Files:")
print(f"{OUTPUT_FILE}")
print(f"{stats_file}")
print()
print("Next Step:")
print("Run Stage 2: python scripts/aggregate_firm_year.py")
print("=" * 80)