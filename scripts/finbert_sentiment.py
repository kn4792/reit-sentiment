"""
Stage 1: FinBERT Sentiment Extraction
======================================
Applies FinBERT sentiment analysis to all REIT Glassdoor reviews.

Input:  data/raw/all_reit_reviews_merged.csv
Output: data/processed/finbert_sentiment_scores.csv

FinBERT: Financial sentiment model fine-tuned on financial news
- Trained on Financial PhraseBank dataset
- Classifies text as positive, negative, or neutral
- Provides probability scores for each class
"""

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

# Configuration
INPUT_FILE = "data/raw/all_reit_reviews_merged.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finbert_sentiment_scores.csv")
BATCH_SIZE = 32  # Process reviews in batches for efficiency
MAX_LENGTH = 512  # FinBERT max sequence length

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("📂 Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ Loaded {len(df):,} reviews")
print(f"  ✓ Columns: {df.columns.tolist()}")
print()

# ============================================================================
# 2. PREPARE TEXT
# ============================================================================
print("📝 Preparing text for sentiment analysis...")

# Combine pros and cons with clear labeling
def prepare_text(row):
    """Combine pros and cons into single text for sentiment analysis."""
    pros = str(row['pros']) if pd.notna(row['pros']) else ""
    cons = str(row['cons']) if pd.notna(row['cons']) else ""
    
    # Label the sections clearly
    if pros and cons:
        return f"Positive aspects: {pros} Negative aspects: {cons}"
    elif pros:
        return f"Positive aspects: {pros}"
    elif cons:
        return f"Negative aspects: {cons}"
    else:
        return ""

df['combined_text'] = df.apply(prepare_text, axis=1)

# Remove empty reviews
df_valid = df[df['combined_text'].str.len() > 0].copy()
print(f"  ✓ Prepared {len(df_valid):,} reviews with valid text")
print(f"  ✓ Excluded {len(df) - len(df_valid):,} empty reviews")
print()

# ============================================================================
# 3. LOAD FINBERT MODEL
# ============================================================================
print("🤖 Loading FinBERT model...")
print("  Model: ProsusAI/finbert")
print()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

print(f"  ✓ Model loaded on: {device}")
print()

# ============================================================================
# 4. RUN SENTIMENT ANALYSIS
# ============================================================================
print(f"🔍 Running sentiment analysis on {len(df_valid):,} reviews...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max sequence length: {MAX_LENGTH}")
print()

# Label mapping
id2label = {0: "positive", 1: "negative", 2: "neutral"}

# Store results
results = []

# Process in batches
texts = df_valid['combined_text'].tolist()
n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

with torch.no_grad():
    for i in tqdm(range(n_batches), desc="Processing batches"):
        # Get batch
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        # Get predictions
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Store results for this batch
        for j, probs in enumerate(predictions):
            probs_cpu = probs.cpu().numpy()
            
            # Get predicted label
            pred_idx = np.argmax(probs_cpu)
            sentiment_label = id2label[pred_idx]
            
            # Create sentiment score: positive - negative (range: -1 to 1)
            sentiment_score = probs_cpu[0] - probs_cpu[1]
            
            results.append({
                'positive_prob': float(probs_cpu[0]),
                'negative_prob': float(probs_cpu[1]),
                'neutral_prob': float(probs_cpu[2]),
                'sentiment_label': sentiment_label,
                'sentiment_score': float(sentiment_score)
            })

print(f"\n  ✓ Sentiment analysis complete!")
print()

# ============================================================================
# 5. COMBINE RESULTS WITH ORIGINAL DATA
# ============================================================================
print("📊 Combining results with original data...")

# Create results dataframe
results_df = pd.DataFrame(results)

# Combine with valid reviews
df_valid = df_valid.reset_index(drop=True)
df_valid = pd.concat([df_valid, results_df], axis=1)

# Add back reviews that were excluded (with NaN sentiment)
df_final = df.merge(
    df_valid[['title', 'date', 'company', 'ticker', 
              'positive_prob', 'negative_prob', 'neutral_prob',
              'sentiment_label', 'sentiment_score']],
    on=['title', 'date', 'company', 'ticker'],
    how='left'
)

print(f"  ✓ Final dataset: {len(df_final):,} reviews")
print()

# ============================================================================
# 6. CALCULATE SUMMARY STATISTICS
# ============================================================================
print("📈 Summary Statistics:")
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
# 7. SAVE RESULTS
# ============================================================================
print("💾 Saving results...")

# Save full dataset
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved: {OUTPUT_FILE}")

# Save summary statistics
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
print(f"  ✓ Saved: {stats_file}")
print()

# ============================================================================
# 8. DISPLAY SAMPLE RESULTS
# ============================================================================
print("🔍 Sample Results:")
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
print("✅ STAGE 1 COMPLETE")
print("=" * 80)
print()
print("📁 Output Files:")
print(f"  • {OUTPUT_FILE}")
print(f"  • {stats_file}")
print()
print("🚀 Next Step:")
print("  Run Stage 2: python scripts/aggregate_firm_year.py")
print("=" * 80)