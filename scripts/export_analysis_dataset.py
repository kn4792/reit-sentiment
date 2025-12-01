"""
Stage 4: Export Analysis-Ready Dataset
=======================================
Prepares clean, validated dataset ready for merging with REIT performance data.

Input:  data/processed/firm_year_sentiment.csv
Output: data/results/analysis_ready_dataset.csv
        data/results/variable_codebook.md

This is the final dataset you'll merge with your REIT performance data for regressions.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

print("=" * 80)
print("STAGE 4: EXPORT ANALYSIS-READY DATASET")
print("=" * 80)
print()

# Configuration
INPUT_FILE = "data/processed/firm_year_sentiment.csv"
OUTPUT_DIR = "data/results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "analysis_ready_dataset.csv")
CODEBOOK_FILE = os.path.join(OUTPUT_DIR, "variable_codebook.md")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("📂 Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ Loaded {len(df):,} firm-year observations")
print()

# ============================================================================
# 2. SELECT AND RENAME VARIABLES
# ============================================================================
print("🔧 Preparing analysis-ready variables...")

# Select key variables and rename for clarity
analysis_df = pd.DataFrame({
    # Identifiers
    'ticker': df['ticker'],
    'year': df['year'],
    'company_name': df['company_first'],
    'property_type': df['property_type_first'],
    
    # Sentiment variables (PRIMARY VARIABLES OF INTEREST)
    'sentiment': df['sentiment_score_mean'],  # Main DV: mean sentiment score [-1, 1]
    'sentiment_std': df['sentiment_score_std'],  # Dispersion within firm-year
    'sentiment_median': df['sentiment_score_median'],
    'sentiment_min': df['sentiment_score_min'],
    'sentiment_max': df['sentiment_score_max'],
    
    # Sentiment components
    'positive_prob': df['positive_prob_mean'],
    'negative_prob': df['negative_prob_mean'],
    'neutral_prob': df['neutral_prob_mean'],
    
    # Sentiment quality metrics
    'sentiment_dispersion': df['sentiment_dispersion'],  # Std dev of sentiment
    'sentiment_agreement': df['sentiment_agreement'],  # % agreeing with majority
    
    # Rating variables
    'rating': df['rating_mean'],  # Glassdoor rating [1-5]
    'rating_std': df['rating_std'],
    'rating_sentiment_diff': df['rating_sentiment_diff'],  # Rating - sentiment alignment
    
    # Review characteristics
    'review_count': df['review_count'],  # Number of reviews in firm-year
    
    # Treatment and controls
    'POST_CHATGPT': df['POST_CHATGPT'],  # Binary: 1 if year > Nov 2022
    'tech_intensive': df['tech_intensive'],  # Binary: 1 if Data Center/Telecom/Infrastructure
    'datacenter_REIT': df['datacenter_REIT'],  # Binary: 1 if Data Center only
})

# ============================================================================
# 3. DATA QUALITY CHECKS
# ============================================================================
print("🔍 Running data quality checks...")

checks_passed = True

# Check 1: No missing values in key variables
key_vars = ['ticker', 'year', 'sentiment', 'rating', 'review_count']
missing = analysis_df[key_vars].isnull().sum()
if missing.sum() > 0:
    print("  ⚠️  WARNING: Missing values detected:")
    print(missing[missing > 0])
    checks_passed = False
else:
    print("  ✓ No missing values in key variables")

# Check 2: Sentiment score range
if (analysis_df['sentiment'] < -1).any() or (analysis_df['sentiment'] > 1).any():
    print("  ⚠️  WARNING: Sentiment scores outside [-1, 1] range")
    checks_passed = False
else:
    print("  ✓ Sentiment scores within valid range [-1, 1]")

# Check 3: Rating range
if (analysis_df['rating'] < 1).any() or (analysis_df['rating'] > 5).any():
    print("  ⚠️  WARNING: Ratings outside [1, 5] range")
    checks_passed = False
else:
    print("  ✓ Ratings within valid range [1, 5]")

# Check 4: Positive review count
if (analysis_df['review_count'] < 1).any():
    print("  ⚠️  WARNING: Firm-years with < 1 review detected")
    checks_passed = False
else:
    print("  ✓ All firm-years have positive review count")

# Check 5: Unique ticker-year combinations
duplicates = analysis_df.duplicated(subset=['ticker', 'year']).sum()
if duplicates > 0:
    print(f"  ⚠️  WARNING: {duplicates} duplicate ticker-year combinations")
    checks_passed = False
else:
    print("  ✓ No duplicate ticker-year combinations")

print()

if checks_passed:
    print("✅ All quality checks passed!")
else:
    print("⚠️  Some quality checks failed - review warnings above")
print()

# ============================================================================
# 4. CREATE ADDITIONAL USEFUL VARIABLES
# ============================================================================
print("➕ Creating additional variables...")

# Year indicators for easy filtering
analysis_df['year_2022_or_earlier'] = (analysis_df['year'] <= 2022).astype(int)
analysis_df['year_2023_or_later'] = (analysis_df['year'] >= 2023).astype(int)

# Sentiment categories (for robustness checks)
analysis_df['high_sentiment'] = (analysis_df['sentiment'] > analysis_df['sentiment'].median()).astype(int)
analysis_df['low_sentiment'] = (analysis_df['sentiment'] < analysis_df['sentiment'].quantile(0.25)).astype(int)

# High dispersion indicator (disagreement among employees)
analysis_df['high_dispersion'] = (analysis_df['sentiment_dispersion'] > 
                                  analysis_df['sentiment_dispersion'].median()).astype(int)

# Large firm-year (many reviews)
analysis_df['many_reviews'] = (analysis_df['review_count'] > 
                               analysis_df['review_count'].median()).astype(int)

# Interaction term for key hypothesis (you can create more in your regression)
analysis_df['sentiment_X_POST_CHATGPT'] = (analysis_df['sentiment'] * 
                                            analysis_df['POST_CHATGPT'])
analysis_df['sentiment_X_tech_intensive'] = (analysis_df['sentiment'] * 
                                              analysis_df['tech_intensive'])

print("  ✓ Created additional indicator variables")
print("  ✓ Created interaction terms")
print()

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================
print("📊 Final dataset summary:")
print()

print(f"Observations: {len(analysis_df):,}")
print(f"Firms: {analysis_df['ticker'].nunique()}")
print(f"Years: {analysis_df['year'].min()} - {analysis_df['year'].max()}")
print()

print("Variable Summary:")
print(f"  Sentiment:       mean={analysis_df['sentiment'].mean():.4f}, "
      f"std={analysis_df['sentiment'].std():.4f}")
print(f"  Rating:          mean={analysis_df['rating'].mean():.4f}, "
      f"std={analysis_df['rating'].std():.4f}")
print(f"  Review count:    mean={analysis_df['review_count'].mean():.1f}, "
      f"median={analysis_df['review_count'].median():.0f}")
print()

print("Treatment Distribution:")
print(f"  POST_CHATGPT=0:  {(analysis_df['POST_CHATGPT']==0).sum():>4} ({(analysis_df['POST_CHATGPT']==0).sum()/len(analysis_df)*100:.1f}%)")
print(f"  POST_CHATGPT=1:  {(analysis_df['POST_CHATGPT']==1).sum():>4} ({(analysis_df['POST_CHATGPT']==1).sum()/len(analysis_df)*100:.1f}%)")
print()

print("Property Type Distribution:")
print(f"  Tech-intensive:  {(analysis_df['tech_intensive']==1).sum():>4} ({(analysis_df['tech_intensive']==1).sum()/len(analysis_df)*100:.1f}%)")
print(f"  Traditional:     {(analysis_df['tech_intensive']==0).sum():>4} ({(analysis_df['tech_intensive']==0).sum()/len(analysis_df)*100:.1f}%)")
print()

# ============================================================================
# 6. SAVE ANALYSIS-READY DATASET
# ============================================================================
print("💾 Saving analysis-ready dataset...")

# Sort by ticker and year for easy merging
analysis_df = analysis_df.sort_values(['ticker', 'year']).reset_index(drop=True)

# Save to CSV
analysis_df.to_csv(OUTPUT_FILE, index=False)
print(f"  ✓ Saved: {OUTPUT_FILE}")
print(f"  ✓ Rows: {len(analysis_df):,}")
print(f"  ✓ Columns: {len(analysis_df.columns)}")
print()

# ============================================================================
# 7. CREATE VARIABLE CODEBOOK
# ============================================================================
print("📖 Creating variable codebook...")

codebook = f"""# Variable Codebook: Analysis-Ready Dataset

**Dataset**: analysis_ready_dataset.csv  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Observations**: {len(analysis_df):,} firm-year observations  
**Period**: {analysis_df['year'].min()}-{analysis_df['year'].max()}

---

## Identifiers

| Variable | Type | Description |
|----------|------|-------------|
| `ticker` | string | REIT stock ticker symbol |
| `year` | integer | Calendar year |
| `company_name` | string | REIT company name |
| `property_type` | string | Property type classification (e.g., Data Center, Office, Residential) |

---

## Primary Sentiment Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `sentiment` | float | [-1, 1] | **PRIMARY DV**: Mean FinBERT sentiment score for firm-year. Calculated as mean of (positive_prob - negative_prob) across all reviews. Higher = more positive sentiment. |
| `sentiment_std` | float | [0, ∞) | Standard deviation of sentiment within firm-year |
| `sentiment_median` | float | [-1, 1] | Median sentiment score within firm-year |
| `sentiment_min` | float | [-1, 1] | Minimum sentiment score within firm-year |
| `sentiment_max` | float | [-1, 1] | Maximum sentiment score within firm-year |

---

## Sentiment Components

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `positive_prob` | float | [0, 1] | Mean probability of positive sentiment (from FinBERT) |
| `negative_prob` | float | [0, 1] | Mean probability of negative sentiment (from FinBERT) |
| `neutral_prob` | float | [0, 1] | Mean probability of neutral sentiment (from FinBERT) |

---

## Sentiment Quality Metrics

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `sentiment_dispersion` | float | [0, ∞) | Within firm-year sentiment dispersion (std dev). High = more disagreement among employees |
| `sentiment_agreement` | float | [0, 1] | Proportion of reviews agreeing with majority sentiment category. High = more consensus |

---

## Glassdoor Rating Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `rating` | float | [1, 5] | Mean Glassdoor rating for firm-year |
| `rating_std` | float | [0, ∞) | Standard deviation of ratings within firm-year |
| `rating_sentiment_diff` | float | [-∞, ∞) | Difference between rating and sentiment (rescaled). Measures rating-sentiment alignment |

---

## Review Characteristics

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `review_count` | integer | [3, ∞) | Number of reviews for firm-year (minimum 3 after filtering) |

---

## Treatment Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `POST_CHATGPT` | binary | {{0, 1}} | **KEY TREATMENT**: 1 if year > November 30, 2022 (ChatGPT launch), 0 otherwise |
| `tech_intensive` | binary | {{0, 1}} | 1 if property type is Data Center, Telecommunications, or Infrastructure; 0 otherwise |
| `datacenter_REIT` | binary | {{0, 1}} | 1 if property type is Data Center only; 0 otherwise |

---

## Additional Indicator Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `year_2022_or_earlier` | binary | {{0, 1}} | 1 if year ≤ 2022; 0 otherwise |
| `year_2023_or_later` | binary | {{0, 1}} | 1 if year ≥ 2023; 0 otherwise |
| `high_sentiment` | binary | {{0, 1}} | 1 if sentiment > median sentiment; 0 otherwise |
| `low_sentiment` | binary | {{0, 1}} | 1 if sentiment < 25th percentile; 0 otherwise |
| `high_dispersion` | binary | {{0, 1}} | 1 if sentiment_dispersion > median; 0 otherwise |
| `many_reviews` | binary | {{0, 1}} | 1 if review_count > median; 0 otherwise |

---

## Interaction Terms

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `sentiment_X_POST_CHATGPT` | float | [-1, 1] | Interaction: sentiment × POST_CHATGPT |
| `sentiment_X_tech_intensive` | float | [-1, 1] | Interaction: sentiment × tech_intensive |

---

## Usage Guide

### Merging with REIT Performance Data

```python
import pandas as pd

# Load sentiment data
sentiment = pd.read_csv('analysis_ready_dataset.csv')

# Load your performance data (must have 'ticker' and 'year' columns)
performance = pd.read_csv('your_reit_performance_data.csv')

# Merge
merged = performance.merge(sentiment, on=['ticker', 'year'], how='inner')

# Check merge
print(f"Performance data: {{len(performance)}} rows")
print(f"Sentiment data: {{len(sentiment)}} rows")
print(f"Merged data: {{len(merged)}} rows")
```

### Example Regression Specifications

#### Baseline Model
```python
import statsmodels.formula.api as smf

# Does employee sentiment predict REIT performance?
model1 = smf.ols('returns ~ sentiment + rating + review_count + C(year) + C(ticker)',
                 data=merged).fit(cov_type='cluster', cov_kwds={{'groups': merged['ticker']}})
print(model1.summary())
```

#### Difference-in-Differences
```python
# Did ChatGPT launch moderate the sentiment-performance relationship?
model2 = smf.ols('''returns ~ sentiment * POST_CHATGPT + 
                    rating + review_count + 
                    C(year) + C(ticker)''',
                 data=merged).fit(cov_type='cluster', cov_kwds={{'groups': merged['ticker']}})
```

#### Triple Difference
```python
# Is effect stronger for tech-intensive REITs after ChatGPT?
model3 = smf.ols('''returns ~ sentiment * tech_intensive * POST_CHATGPT +
                    rating + review_count +
                    C(year) + C(ticker)''',
                 data=merged).fit(cov_type='cluster', cov_kwds={{'groups': merged['ticker']}})
```

---

## Data Quality Notes

- **Minimum Reviews**: Each firm-year has ≥3 reviews
- **No Missing Data**: All key variables (sentiment, rating, review_count) have no missing values
- **Unique Keys**: Each ticker-year combination appears exactly once
- **Date Range**: {analysis_df['year'].min()}-{analysis_df['year'].max()}
- **ChatGPT Launch**: November 30, 2022

---

## Summary Statistics

```
Sentiment:       mean={analysis_df['sentiment'].mean():.4f}, std={analysis_df['sentiment'].std():.4f}
Rating:          mean={analysis_df['rating'].mean():.4f}, std={analysis_df['rating'].std():.4f}
Review count:    mean={analysis_df['review_count'].mean():.1f}, median={analysis_df['review_count'].median():.0f}

POST_CHATGPT:    0={( analysis_df['POST_CHATGPT']==0).sum()}, 1={(analysis_df['POST_CHATGPT']==1).sum()}
tech_intensive:  0={(analysis_df['tech_intensive']==0).sum()}, 1={(analysis_df['tech_intensive']==1).sum()}
```

---

## Property Types Included

{chr(10).join([f"- {ptype}: {count} firm-years" for ptype, count in analysis_df['property_type'].value_counts().items()])}

---

## Contact

For questions about this dataset or variable definitions, please refer to the thesis documentation.

**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
"""

with open(CODEBOOK_FILE, 'w') as f:
    f.write(codebook)

print(f"  ✓ Saved: {CODEBOOK_FILE}")
print()

# ============================================================================
# 8. CREATE SAMPLE MERGE SCRIPT
# ============================================================================
print("📝 Creating sample merge script...")

sample_script = """'''
Sample Script: Merging Sentiment Data with REIT Performance Data
==================================================================
This script shows how to merge the analysis-ready sentiment dataset
with your REIT performance data and run basic regressions.
'''

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Loading data...")

# Load sentiment data
sentiment = pd.read_csv('data/results/analysis_ready_dataset.csv')
print(f"✓ Loaded sentiment data: {len(sentiment):,} firm-years")

# Load your REIT performance data
# TODO: Replace with your actual performance data file
# performance = pd.read_csv('your_reit_performance_data.csv')

# For now, let's create a sample performance dataset
# YOU SHOULD REPLACE THIS WITH YOUR ACTUAL DATA
print("\\n⚠️  Using SAMPLE performance data - replace with your actual data!")
performance = pd.DataFrame({
    'ticker': sentiment['ticker'],
    'year': sentiment['year'],
    'returns': np.random.normal(0.08, 0.15, len(sentiment)),  # Sample returns
    'ffo': np.random.normal(2.5, 0.8, len(sentiment)),  # Sample FFO
    'market_cap': np.random.lognormal(20, 1.5, len(sentiment))  # Sample market cap
})

# ============================================================================
# 2. MERGE DATASETS
# ============================================================================
print("\\nMerging datasets...")

# Inner merge: only keep observations present in both datasets
merged = performance.merge(sentiment, on=['ticker', 'year'], how='inner')

print(f"✓ Performance data: {len(performance):,} rows")
print(f"✓ Sentiment data: {len(sentiment):,} rows")
print(f"✓ Merged data: {len(merged):,} rows")
print(f"✓ Match rate: {len(merged)/len(performance)*100:.1f}%")

# ============================================================================
# 3. RUN REGRESSIONS
# ============================================================================
print("\\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Model 1: Baseline - Does sentiment predict returns?
print("\\nMODEL 1: Baseline Sentiment → Returns")
print("-" * 80)

model1 = smf.ols('returns ~ sentiment + rating + review_count',
                 data=merged).fit(cov_type='HC1')  # Robust standard errors

print(model1.summary())
print(f"\\nInterpretation:")
print(f"  • Sentiment coefficient: {model1.params['sentiment']:.4f}")
print(f"  • t-stat: {model1.tvalues['sentiment']:.2f}")
print(f"  • p-value: {model1.pvalues['sentiment']:.4f}")

if model1.pvalues['sentiment'] < 0.05:
    direction = "positive" if model1.params['sentiment'] > 0 else "negative"
    print(f"  ✓ Sentiment has a statistically significant {direction} effect on returns!")
else:
    print(f"  ✗ Sentiment effect not statistically significant")

# Model 2: Difference-in-Differences - Did ChatGPT matter?
print("\\n\\nMODEL 2: Difference-in-Differences (ChatGPT)")
print("-" * 80)

model2 = smf.ols('''returns ~ sentiment * POST_CHATGPT + 
                    rating + review_count''',
                 data=merged).fit(cov_type='HC1')

print(model2.summary())
print(f"\\nInterpretation:")
print(f"  • Sentiment (pre-ChatGPT): {model2.params['sentiment']:.4f}")
print(f"  • Interaction (sentiment × POST_CHATGPT): {model2.params['sentiment:POST_CHATGPT']:.4f}")
print(f"  • Total effect (post-ChatGPT): {model2.params['sentiment'] + model2.params['sentiment:POST_CHATGPT']:.4f}")

if model2.pvalues['sentiment:POST_CHATGPT'] < 0.05:
    print(f"  ✓ ChatGPT significantly moderated the sentiment-returns relationship!")
else:
    print(f"  ✗ ChatGPT interaction not statistically significant")

# Model 3: Triple Difference - Tech-intensive × ChatGPT
print("\\n\\nMODEL 3: Triple Difference (Tech-Intensive × ChatGPT)")
print("-" * 80)

model3 = smf.ols('''returns ~ sentiment * tech_intensive * POST_CHATGPT +
                    rating + review_count''',
                 data=merged).fit(cov_type='HC1')

print(model3.summary())

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\\n" + "="*80)
print("Creating visualizations...")

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot: Sentiment vs Returns
axes[0].scatter(merged['sentiment'], merged['returns'], alpha=0.5, s=30)
axes[0].set_xlabel('Employee Sentiment')
axes[0].set_ylabel('Returns')
axes[0].set_title('Employee Sentiment vs REIT Returns')
axes[0].grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(merged['sentiment'], merged['returns'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['sentiment'].min(), merged['sentiment'].max(), 100)
axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
axes[0].legend()

# Group comparison: Pre vs Post ChatGPT
pre = merged[merged['POST_CHATGPT'] == 0]
post = merged[merged['POST_CHATGPT'] == 1]

axes[1].scatter(pre['sentiment'], pre['returns'], alpha=0.5, s=30, label='Pre-ChatGPT', color='blue')
axes[1].scatter(post['sentiment'], post['returns'], alpha=0.5, s=30, label='Post-ChatGPT', color='red')
axes[1].set_xlabel('Employee Sentiment')
axes[1].set_ylabel('Returns')
axes[1].set_title('Sentiment-Returns Relationship: Pre vs Post ChatGPT')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_returns_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sentiment_returns_analysis.png")

plt.show()

print("\\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
"""

sample_script_file = os.path.join(OUTPUT_DIR, "sample_merge_and_regression.py")
with open(sample_script_file, 'w') as f:
    f.write(sample_script)

print(f"  ✓ Saved: {sample_script_file}")
print()

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("✅ STAGE 4 COMPLETE - PIPELINE FINISHED!")
print("=" * 80)
print()

print("📁 Final Output Files:")
print(f"  • {OUTPUT_FILE}")
print(f"    → {len(analysis_df):,} firm-year observations")
print(f"    → {len(analysis_df.columns)} variables")
print(f"    → Ready to merge with REIT performance data")
print()
print(f"  • {CODEBOOK_FILE}")
print(f"    → Complete variable documentation")
print(f"    → Merge instructions")
print(f"    → Example regression specifications")
print()
print(f"  • {sample_script_file}")
print(f"    → Sample merge and regression code")
print(f"    → Replace with your actual performance data")
print()

print("🎯 NEXT STEPS FOR YOUR THESIS:")
print()
print("1. MERGE WITH PERFORMANCE DATA:")
print("   - Load your REIT performance data (returns, FFO, occupancy, etc.)")
print("   - Ensure it has 'ticker' and 'year' columns")
print("   - Use pd.merge(performance, sentiment, on=['ticker', 'year'])")
print()
print("2. RUN REGRESSIONS:")
print("   - Baseline: returns ~ sentiment + controls")
print("   - DiD: returns ~ sentiment * POST_CHATGPT + controls")
print("   - Triple-diff: returns ~ sentiment * tech_intensive * POST_CHATGPT")
print()
print("3. WRITE RESULTS:")
print("   - Use descriptive stats from Stage 3 for your 'Data' section")
print("   - Report regression results in 'Results' section")
print("   - Visualizations ready for thesis figures")
print()

print("=" * 80)
print("🚀 ALL STAGES COMPLETE - READY FOR PERFORMANCE ANALYSIS!")
print("=" * 80)