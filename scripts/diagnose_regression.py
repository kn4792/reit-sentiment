#!/usr/bin/env python3
"""
Diagnostic script to understand regression convergence issues.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

# Load data
print("Loading data...")
firm_year_df = pd.read_csv('data/processed/mnir/firm_year_data.csv')
word_counts_pros = pd.read_csv('data/processed/mnir/word_counts_pros.csv')

print(f"\nFirm-year shape: {firm_year_df.shape}")
print(f"Word counts shape: {word_counts_pros.shape}")

# Merge
df = firm_year_df.merge(word_counts_pros, on=['ticker', 'year'], how='inner')
print(f"\nMerged shape: {df.shape}")

# Check genai_intensity
print(f"\n{'='*70}")
print("GenAI Intensity Distribution:")
print(f"{'='*70}")
print(df['genai_intensity'].describe())
print(f"\nUnique values: {df['genai_intensity'].nunique()}")
print(f"Zeros: {(df['genai_intensity'] == 0).sum()} / {len(df)}")

# Prepare covariates
df['intercept'] = 1.0
df['log_reviews'] = np.log1p(df['review_count'])

# Add year FE
year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True)
X = pd.concat([
    df[['intercept', 'rating', 'log_reviews', 'genai_intensity']],
    year_dummies
], axis=1)

print(f"\n{'='*70}")
print("Covariate Matrix:")
print(f"{'='*70}")
print(f"Shape: {X.shape}")
print(f"\nColumn names: {X.columns.tolist()[:10]}...")
print(f"\nChecking for NaNs in X:")
print(X.isna().sum().sum())

print(f"\nChecking variance in key variables:")
for col in ['rating', 'log_reviews', 'genai_intensity']:
    print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, var={df[col].var():.6f}")

# Try a single regression
print(f"\n{'='*70}")
print("Testing Single Regression:")
print(f"{'='*70}")

# Pick first word column
word_cols = [c for c in df.columns if c.startswith('wc_')]
test_word_col = word_cols[100]  # Try the 100th word
test_word = test_word_col[3:]

y = df[test_word_col].values
print(f"\nTest word: {test_word}")
print(f"  Counts - mean: {y.mean():.4f}, std: {y.std():.4f}, max: {y.max()}")
print(f"  Zero counts: {(y == 0).sum()} / {len(y)}")

# Create offset
offset = np.log(df['word_count_pros'] + 1).values
print(f"\nOffset - mean: {offset.mean():.4f}, std: {offset.std():.4f}")

# Try regression
print(f"\nAttempting Poisson regression...")
try:
    model = sm.GLM(y, X, family=Poisson(), offset=offset)
    result = model.fit(maxiter=100, disp=True)
    
    print(f"\nConverged: {result.converged}")
    print(f"\nGenAI Intensity coefficient:")
    print(f"  Coef: {result.params['genai_intensity']:.6f}")
    print(f"  T-stat: {result.tvalues['genai_intensity']:.4f}")
    print(f"  P-value: {result.pvalues['genai_intensity']:.4f}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Check for multicollinearity
print(f"\n{'='*70}")
print("Checking for Multicollinearity:")
print(f"{'='*70}")

from numpy.linalg import matrix_rank
print(f"X rank: {matrix_rank(X.values)}")
print(f"X shape: {X.shape}")
print(f"Full rank? {matrix_rank(X.values) == X.shape[1]}")

# Check correlation between genai_intensity and other vars
print(f"\nCorrelations with genai_intensity:")
for col in ['rating', 'log_reviews', 'word_count_pros']:
    if col in df.columns:
        corr = df['genai_intensity'].corr(df[col])
        print(f"  {col}: {corr:.4f}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)