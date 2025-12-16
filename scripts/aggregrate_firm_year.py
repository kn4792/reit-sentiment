## @file aggregrate_firm_year.py
# @brief Stage 2: Aggregate to Firm-Year Level
#
# Aggregates review-level sentiment scores to firm-year level for panel analysis.
#
# @details
# Input:  data/processed/finbert_sentiment_scores.csv
# Output: data/processed/firm_year_sentiment.csv
#
# This script performs the following operations:
# - Groups reviews by ticker and year
# - Calculates aggregate sentiment metrics (mean, std, median)
# - Creates POST_CHATGPT treatment indicator
# - Classifies property types (tech-intensive, datacenter)
# - Applies quality filters (minimum reviews per firm-year)
#
# @author Konain Niaz (kn4792@rit.edu)
# @date 2025-12-16
# @version 1.0

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from scipy import stats

print("=" * 80)
print("STAGE 2: AGGREGATE TO FIRM-YEAR LEVEL")
print("=" * 80)
print()

## @var INPUT_FILE
# Path to input CSV file with review-level sentiment scores
INPUT_FILE = "data/processed/finbert_sentiment_scores.csv"

## @var OUTPUT_DIR
# Directory for output files
OUTPUT_DIR = "data/processed"

## @var OUTPUT_FILE
# Path to output CSV file with firm-year aggregated data
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "firm_year_sentiment.csv")

## @var CHATGPT_LAUNCH
# Date of ChatGPT public launch (treatment date for difference-in-differences)
CHATGPT_LAUNCH = "2022-11-30"

# ============================================================================
# STEP 1: LOAD DATA
# Load the sentiment scores from Stage 1
# ============================================================================
print("Loading sentiment data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} reviews")
print()

# ============================================================================
# STEP 2: PREPARE VARIABLES
# Parse dates and create treatment indicators
# ============================================================================
print("Preparing variables...")

# Parse date column to datetime format
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Create POST_CHATGPT binary indicator for DiD analysis
# POST_CHATGPT = 1 if review date is after Nov 30, 2022
chatgpt_date = pd.to_datetime(CHATGPT_LAUNCH)
df['POST_CHATGPT'] = (df['date'] > chatgpt_date).astype(int)

print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Years: {df['year'].min()} to {df['year'].max()}")
print(f"Pre-ChatGPT reviews: {(df['POST_CHATGPT'] == 0).sum():,}")
print(f"Post-ChatGPT reviews: {(df['POST_CHATGPT'] == 1).sum():,}")
print()

# ============================================================================
# STEP 3: PROPERTY TYPE CLASSIFICATION
# Identify technology-intensive REITs for heterogeneity analysis
# ============================================================================
print("Classifying property types...")

## @var TECH_INTENSIVE_TYPES
# List of property types considered technology-intensive
TECH_INTENSIVE_TYPES = ['Data Center', 'Telecommunications', 'Infrastructure']

# Create binary indicator for tech-intensive REITs
df['tech_intensive'] = df['property_type'].isin(TECH_INTENSIVE_TYPES).astype(int)

# Create specific indicator for data center REITs
df['datacenter_REIT'] = (df['property_type'] == 'Data Center').astype(int)

# Display distribution
print("Property Type Distribution:")
prop_dist = df['property_type'].value_counts()
for ptype, count in prop_dist.items():
    indicator = "[TECH]" if ptype in TECH_INTENSIVE_TYPES else "  "
    print(f"    {indicator} {ptype:<25} {count:>6,}")
print()

print(f"Technology-intensive REITs: {df['tech_intensive'].sum():,} reviews")
print(f"Data Center REITs: {df['datacenter_REIT'].sum():,} reviews")
print()

# ============================================================================
# STEP 4: AGGREGATE TO FIRM-YEAR
# Group by ticker and year, calculate aggregate metrics
# ============================================================================
print("Aggregating to firm-year level...")

# Filter to reviews with valid sentiment scores
df_valid = df[df['sentiment_score'].notna()].copy()
print(f"Using {len(df_valid):,} reviews with valid sentiment")

## @var agg_functions
# Dictionary defining aggregation functions for each column
agg_functions = {
    # Sentiment metrics - calculate multiple statistics
    'sentiment_score': ['mean', 'std', 'median', 'min', 'max'],
    'positive_prob': ['mean', 'std'],
    'negative_prob': ['mean', 'std'],
    'neutral_prob': ['mean', 'std'],
    
    # Review characteristics
    'rating': ['mean', 'std', 'count'],
    
    # Treatment and controls - take first value (same within group)
    'POST_CHATGPT': 'first',
    'property_type': 'first',
    'tech_intensive': 'first',
    'datacenter_REIT': 'first',
    'company': 'first',
}

# Group by ticker and year, apply aggregation functions
firm_year = df_valid.groupby(['ticker', 'year']).agg(agg_functions)

# Flatten multi-level column names
firm_year.columns = ['_'.join(col).strip('_') for col in firm_year.columns.values]

# Rename columns for clarity
rename_map = {
    'rating_count': 'review_count',
    'POST_CHATGPT_first': 'POST_CHATGPT',
    'property_type_first': 'property_type',
    'tech_intensive_first': 'tech_intensive',
    'datacenter_REIT_first': 'datacenter_REIT',
    'company_first': 'company',
}
firm_year.rename(columns=rename_map, inplace=True)

# Reset index to make ticker and year regular columns
firm_year = firm_year.reset_index()

print(f"Created {len(firm_year):,} firm-year observations")
print(f"Unique firms: {firm_year['ticker'].nunique()}")
print(f"Year range: {firm_year['year'].min()} to {firm_year['year'].max()}")
print()

# ============================================================================
# STEP 5: CALCULATE ADDITIONAL METRICS
# Compute sentiment dispersion and agreement measures
# ============================================================================
print("Calculating additional metrics...")

# Sentiment dispersion: within firm-year standard deviation
firm_year['sentiment_dispersion'] = firm_year['sentiment_score_std'].fillna(0)


## Calculate percentage of reviews agreeing with majority sentiment.
#
# This function computes the proportion of reviews within a firm-year
# that share the same sentiment label as the majority of reviews.
# Higher agreement indicates more consistent employee sentiment.
#
# @param group A pandas DataFrame group containing reviews for one firm-year
# @return Float between 0 and 1 representing agreement ratio
def calculate_agreement(group):
    if len(group) < 2:
        return 1.0
    
    # Count each sentiment category
    pos = (group['sentiment_label'] == 'positive').sum()
    neg = (group['sentiment_label'] == 'negative').sum()
    neu = (group['sentiment_label'] == 'neutral').sum()
    
    # Find majority count and calculate ratio
    majority = max(pos, neg, neu)
    return majority / len(group)


# Apply agreement calculation to each firm-year group
agreement = df_valid.groupby(['ticker', 'year']).apply(calculate_agreement)
agreement = agreement.reset_index()
agreement.columns = ['ticker', 'year', 'sentiment_agreement']

# Merge agreement scores back to firm-year data
firm_year = firm_year.merge(agreement, on=['ticker', 'year'], how='left')

# Calculate rating-sentiment alignment
# Transforms sentiment score to 0-5 scale for comparison with rating
firm_year['rating_sentiment_diff'] = firm_year['rating_mean'] - (
    (firm_year['sentiment_score_mean'] + 1) / 2 * 5
)

print("Sentiment dispersion calculated")
print("Sentiment agreement calculated")
print("Rating-sentiment alignment calculated")
print()

# ============================================================================
# STEP 6: QUALITY FILTERS
# Require minimum number of reviews for reliable estimates
# ============================================================================
print("Applying quality filters...")

initial_count = len(firm_year)

## @var MIN_REVIEWS
# Minimum number of reviews required per firm-year observation
MIN_REVIEWS = 3

# Filter out firm-years with insufficient reviews
firm_year = firm_year[firm_year['review_count'] >= MIN_REVIEWS].copy()

print(f"Minimum reviews per firm-year: {MIN_REVIEWS}")
print(f"Filtered: {initial_count:,} → {len(firm_year):,} firm-years")
print(f"Excluded: {initial_count - len(firm_year):,} firm-years with <{MIN_REVIEWS} reviews")
print()

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# Display descriptive statistics for the aggregated dataset
# ============================================================================
print("Summary Statistics:")
print()

# Overall statistics
print("Firm-Year Dataset:")
print(f"Observations: {len(firm_year):,}")
print(f"Firms: {firm_year['ticker'].nunique()}")
print(f"Years: {firm_year['year'].min()} - {firm_year['year'].max()}")
print(f"Avg reviews per firm-year: {firm_year['review_count'].mean():.1f}")
print()

# Pre/Post ChatGPT distribution
print("ChatGPT Treatment Distribution:")
pre_count = (firm_year['POST_CHATGPT'] == 0).sum()
post_count = (firm_year['POST_CHATGPT'] == 1).sum()
print(f"Pre-ChatGPT:  {pre_count:>4} firm-years ({pre_count/len(firm_year)*100:.1f}%)")
print(f"Post-ChatGPT: {post_count:>4} firm-years ({post_count/len(firm_year)*100:.1f}%)")
print()

# Property type distribution
print("Property Type Distribution (Firm-Years):")
for ptype, count in firm_year['property_type'].value_counts().items():
    pct = count / len(firm_year) * 100
    print(f"{ptype:<25} {count:>4} ({pct:>5.1f}%)")
print()

# Sentiment statistics
print("Sentiment Statistics:")
print(f"Mean sentiment:      {firm_year['sentiment_score_mean'].mean():>7.4f}")
print(f"  Median sentiment:    {firm_year['sentiment_score_mean'].median():>7.4f}")
print(f"  Mean dispersion:     {firm_year['sentiment_dispersion'].mean():>7.4f}")
print(f"  Mean agreement:      {firm_year['sentiment_agreement'].mean():>7.4f}")
print()

# Sentiment by property type
print("Sentiment by Property Type:")
for ptype in ['Data Center', 'Office', 'Residential', 'Retail', 'Industrial']:
    subset = firm_year[firm_year['property_type'] == ptype]
    if len(subset) > 0:
        mean_sent = subset['sentiment_score_mean'].mean()
        print(f"{ptype:<25} {mean_sent:>7.4f} (n={len(subset)})")
print()

# Pre/Post ChatGPT comparison with t-test
print("Sentiment: Pre vs Post ChatGPT")
pre_sent = firm_year[firm_year['POST_CHATGPT'] == 0]['sentiment_score_mean'].mean()
post_sent = firm_year[firm_year['POST_CHATGPT'] == 1]['sentiment_score_mean'].mean()
diff = post_sent - pre_sent
print(f"Pre-ChatGPT:  {pre_sent:>7.4f}")
print(f"Post-ChatGPT: {post_sent:>7.4f}")
print(f"Difference:   {diff:>7.4f} ({diff/abs(pre_sent)*100:+.1f}%)")

# Perform independent samples t-test
pre_data = firm_year[firm_year['POST_CHATGPT'] == 0]['sentiment_score_mean']
post_data = firm_year[firm_year['POST_CHATGPT'] == 1]['sentiment_score_mean']
t_stat, p_value = stats.ttest_ind(pre_data, post_data)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print(f"Statistically significant difference!")
else:
    print(f"Not statistically significant")
print()

# ============================================================================
# STEP 8: SAVE RESULTS
# Export aggregated data and summary statistics
# ============================================================================
print("Saving results...")

# Save firm-year dataset
os.makedirs(OUTPUT_DIR, exist_ok=True)
firm_year.to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")

# Save summary statistics as JSON
summary_stats = {
    'n_firm_years': len(firm_year),
    'n_firms': int(firm_year['ticker'].nunique()),
    'year_range': [int(firm_year['year'].min()), int(firm_year['year'].max())],
    'pre_chatgpt': int(pre_count),
    'post_chatgpt': int(post_count),
    'avg_reviews_per_firm_year': float(firm_year['review_count'].mean()),
    'property_type_distribution': firm_year['property_type'].value_counts().to_dict(),
    'sentiment_stats': {
        'mean': float(firm_year['sentiment_score_mean'].mean()),
        'median': float(firm_year['sentiment_score_mean'].median()),
        'std': float(firm_year['sentiment_score_mean'].std()),
        'pre_chatgpt': float(pre_sent),
        'post_chatgpt': float(post_sent),
        'difference': float(diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value)
    },
    'timestamp': datetime.now().isoformat()
}

stats_file = os.path.join(OUTPUT_DIR, "firm_year_summary_stats.json")
with open(stats_file, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"Saved: {stats_file}")
print()

print("=" * 80)
print("STAGE 2 COMPLETE")
print("=" * 80)
print()
print("Output Files:")
print(f"• {OUTPUT_FILE}")
print(f"• {stats_file}")
print()
print("Next Step:")
print("Run Stage 3: python scripts/descriptive_analysis.py")
print("=" * 80)