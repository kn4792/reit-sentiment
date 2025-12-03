#!/usr/bin/env python3
"""
Map Stemmed Words to Original Words for Display

Builds a mapping from Porter stemmed words back to their most common
original forms, making MNIR results more interpretable.

This script:
1. Loads raw reviews to build stem -> original word mapping
2. Identifies most frequent original word for each stem
3. Updates word_weights.csv with readable 'original_word' column
4. Creates top 50 words file with both stems and original words

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-11-29

Usage:
    python scripts/map_stems_to_words.py
"""

import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import re
from pathlib import Path

print("="*70)
print("STEM-TO-WORD MAPPING")
print("="*70)

# Initialize stemmer
stemmer = PorterStemmer()

# Load raw reviews to build mapping
print("\nLoading raw reviews...")
input_file = Path('data/raw/all_reit_reviews_merged.csv')

if not input_file.exists():
    print(f"Error: {input_file} not found")
    print("   Make sure you're running from the project root directory")
    exit(1)

df = pd.read_csv(input_file)
print(f"Loaded {len(df):,} reviews")

# Build stem -> original word mapping
print("\nBuilding stem-to-word mapping...")
stem_to_words = defaultdict(Counter)

for col in ['pros', 'cons']:
    print(f"  Processing {col}...")
    for text in df[col].fillna(''):
        if not text:
            continue
        
        # Clean but don't stem (same cleaning as preprocessing)
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        words = text.split()
        words = [w for w in words if len(w) >= 3]
        
        # Map each word to its stem
        for word in words:
            stem = stemmer.stem(word)
            stem_to_words[stem][word] += 1

# Create mapping (stem -> most common original word)
print("\nCreating stem -> word mapping...")
stem_mapping = {}
for stem, word_counts in stem_to_words.items():
    most_common_word = word_counts.most_common(1)[0][0]
    stem_mapping[stem] = most_common_word

print(f"Created mapping for {len(stem_mapping):,} stems")

# Load word weights
print("\nLoading word weights...")
weights_file = Path('data/results/mnir/word_weights.csv')

if not weights_file.exists():
    print(f"Error: {weights_file} not found")
    print("   Run mnir_regression.py first to generate word weights")
    exit(1)

weights = pd.read_csv(weights_file)
print(f"Loaded {len(weights):,} word weights")

# Add original word column
print("\nMapping stems to original words...")
weights['original_word'] = weights['word'].map(stem_mapping)

# Fill any missing mappings with the stem itself
weights['original_word'] = weights['original_word'].fillna(weights['word'])

# Reorder columns for readability
weights = weights[['original_word', 'word', 'section', 'coef', 't_stat', 'p_value', 
                   'converged', 'mean_count', 'std_count', 'status']]

# Save updated weights
output_file = Path('data/results/mnir/word_weights_readable.csv')
weights.to_csv(output_file, index=False)
print(f"Saved readable weights to: {output_file}")

# Create top 50 with readable words
print("\nCreating top 50 words file...")
top_50 = weights[
    (weights['converged'] == True) & 
    (weights['t_stat'].abs() >= 1.96)
].sort_values('coef', ascending=False).head(50)

top_50_file = Path('data/results/mnir/top_50_chatgpt_words_readable.csv')
top_50[['original_word', 'word', 'section', 'coef', 't_stat', 'p_value']].to_csv(
    top_50_file, index=False
)
print(f"Saved top 50 readable words to: {top_50_file}")

# Print sample mappings
print("\n" + "="*70)
print("SAMPLE MAPPINGS")
print("="*70)
print("\nStem               -> Original Word")
print("-" * 50)

# Show some interesting examples
example_stems = []
for stem in ['autom', 'automat', 'machin', 'technolog', 'digit', 
             'platform', 'effici', 'product', 'innov', 'adapt']:
    if stem in stem_mapping:
        example_stems.append(stem)

for stem in example_stems[:10]:
    original = stem_mapping[stem]
    print(f"{stem:18} -> {original}")

# Show top weighted words with both forms
print("\n" + "="*70)
print("TOP 10 POST-CHATGPT WORDS (with original forms)")
print("="*70)

top_10 = weights[
    (weights['converged'] == True) & 
    (weights['t_stat'].abs() >= 1.96) &
    (weights['coef'] > 0)  # Words that increased post-ChatGPT
].sort_values('coef', ascending=False).head(10)

if len(top_10) > 0:
    print("\nOriginal Word      Stem               Section    Coefficient")
    print("-" * 70)
    for idx, row in top_10.iterrows():
        print(f"{row['original_word']:18} {row['word']:18} {row['section']:10} {row['coef']:8.4f}")
else:
    print("No significant words with positive coefficients found")

print("\n" + "="*70)
print("MAPPING COMPLETE")
print("="*70)
print("\nOutput files:")
print(f"  {output_file.name}")
print(f"  {top_50_file.name}")
print("\nNext step:")
print("  Run: python scripts/analysis.py")
print("="*70)