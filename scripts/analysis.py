#!/usr/bin/env python3
"""
ChatGPT Language Index Analysis

Analyzes the MNIR-derived ChatGPT Language Index and prepares it for
merging with REIT performance data.

The index measures: "To what extent do firm-year reviews use language patterns
that emerged after ChatGPT's launch (Nov 30, 2022)?"

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-11-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("CHATGPT LANGUAGE INDEX ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*70)
print("1. LOADING DATA")
print("="*70)

index_df = pd.read_csv('data/results/mnir/chatgpt_language_index.csv')
weights_df = pd.read_csv('data/results/mnir/word_weights_readable.csv')
firm_year_df = pd.read_csv('data/processed/mnir/firm_year_data.csv')

print(f"\n✓ ChatGPT Language Index: {len(index_df):,} firm-years")
print(f"✓ Word Weights: {len(weights_df):,} words")
print(f"✓ Firm-Year Data: {len(firm_year_df):,} observations")

# Merge index with firm-year data for controls
df = index_df.merge(firm_year_df, on=['ticker', 'year'], how='left')

print(f"\n✓ Merged dataset: {len(df):,} firm-years")
print(f"  Columns: {', '.join(df.columns)}")

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*70)
print("2. DESCRIPTIVE STATISTICS")
print("="*70)

print("\nChatGPT Language Index Statistics:")
print(df['chatgpt_language_index'].describe())

print("\nDistribution by Quartile:")
df['index_quartile'] = pd.qcut(df['chatgpt_language_index'], 
                                q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
print(df['index_quartile'].value_counts().sort_index())

print("\nTop 10 Firms by Average ChatGPT Language Index:")
top_firms = df.groupby('ticker')['chatgpt_language_index'].mean().sort_values(ascending=False).head(10)
for i, (ticker, score) in enumerate(top_firms.items(), 1):
    print(f"  {i:2d}. {ticker}: {score:7.4f}")

print("\nBottom 10 Firms by Average ChatGPT Language Index:")
bottom_firms = df.groupby('ticker')['chatgpt_language_index'].mean().sort_values(ascending=True).head(10)
for i, (ticker, score) in enumerate(bottom_firms.items(), 1):
    print(f"  {i:2d}. {ticker}: {score:7.4f}")

# ============================================================================
# 3. TIME TRENDS
# ============================================================================
print("\n" + "="*70)
print("3. TIME TRENDS")
print("="*70)

# Average by year
yearly_avg = df.groupby('year')['chatgpt_language_index'].agg(['mean', 'std', 'count'])
print("\nAverage ChatGPT Language Index by Year:")
print(yearly_avg)

# Pre vs Post ChatGPT comparison
pre_mean = df[df['POST_CHATGPT'] == 0]['chatgpt_language_index'].mean()
post_mean = df[df['POST_CHATGPT'] == 1]['chatgpt_language_index'].mean()
diff = post_mean - pre_mean

print(f"\nPre-ChatGPT (≤Nov 30, 2022): {pre_mean:.6f}")
print(f"Post-ChatGPT (>Nov 30, 2022): {post_mean:.6f}")
print(f"Difference: {diff:.6f} ({diff/abs(pre_mean)*100:.2f}% change)")

# Statistical test
pre_data = df[df['POST_CHATGPT'] == 0]['chatgpt_language_index']
post_data = df[df['POST_CHATGPT'] == 1]['chatgpt_language_index']
t_stat, p_val = stats.ttest_ind(post_data, pre_data)
print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")

if p_val < 0.05:
    print("✓ Statistically significant difference!")
else:
    print("⚠️  Not statistically significant at p<0.05")

# ============================================================================
# 4. TOP PREDICTIVE WORDS
# ============================================================================
print("\n" + "="*70)
print("4. TOP PREDICTIVE WORDS (POST-CHATGPT LANGUAGE)")
print("="*70)

# Filter significant words
sig_weights = weights_df[
    (weights_df['converged'] == True) & 
    (weights_df['t_stat'].abs() >= 1.96)
].copy()

print(f"\nTotal significant words: {len(sig_weights):,}")

# Top words by section (positive association = increased post-ChatGPT)
for section in ['pros', 'cons']:
    print(f"\n{section.upper()} - Top 20 Words (Increased Post-ChatGPT):")
    top_words = sig_weights[sig_weights['section'] == section].sort_values(
        'coef', ascending=False
    ).head(20)
    
    for i, row in enumerate(top_words.itertuples(), 1):
        print(f"  {i:2d}. {row.original_word:<20} (stem: {row.word:<15}) coef: {row.coef:7.4f}  t: {row.t_stat:7.2f}")

    print(f"\n{section.upper()} - Top 20 Words (Decreased Post-ChatGPT):")
    bottom_words = sig_weights[sig_weights['section'] == section].sort_values(
        'coef', ascending=True
    ).head(20)
    
    for i, row in enumerate(bottom_words.itertuples(), 1):
        print(f"  {i:2d}. {row.original_word:<20} (stem: {row.word:<15}) coef: {row.coef:7.4f}  t: {row.t_stat:7.2f}")

# ============================================================================
# 5. CORRELATIONS WITH REVIEW CHARACTERISTICS
# ============================================================================
print("\n" + "="*70)
print("5. CORRELATIONS WITH REVIEW CHARACTERISTICS")
print("="*70)

correlations = df[['chatgpt_language_index', 'rating', 'review_count', 
                    'POST_CHATGPT', 'word_count_pros', 'word_count_cons']].corr()

print("\nCorrelations with ChatGPT Language Index:")
print(correlations['chatgpt_language_index'].sort_values(ascending=False))

# ============================================================================
# 6. CREATING VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("6. CREATING VISUALIZATIONS")
print("="*70)

output_dir = Path('data/results/mnir/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['chatgpt_language_index'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(df['chatgpt_language_index'].mean(), color='red', linestyle='--', 
           label=f'Mean = {df["chatgpt_language_index"].mean():.4f}')
ax.axvline(df['chatgpt_language_index'].median(), color='green', linestyle='--',
           label=f'Median = {df["chatgpt_language_index"].median():.4f}')
ax.set_xlabel('ChatGPT Language Index', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of ChatGPT Language Index', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'index_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: index_distribution.png")
plt.close()

# Plot 2: Time Trends
fig, ax = plt.subplots(figsize=(12, 6))
yearly_avg['mean'].plot(ax=ax, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.axvline(2022.9, color='red', linestyle='--', linewidth=2, 
           label='ChatGPT Launch (Nov 30, 2022)')
ax.fill_between(yearly_avg.index, 
                yearly_avg['mean'] - yearly_avg['std'], 
                yearly_avg['mean'] + yearly_avg['std'],
                alpha=0.3, color='steelblue')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Average ChatGPT Language Index', fontsize=12)
ax.set_title('ChatGPT Language Index Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'index_time_trend.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: index_time_trend.png")
plt.close()

# Plot 3: Pre vs Post ChatGPT comparison
fig, ax = plt.subplots(figsize=(10, 6))
pre_post_data = df.groupby('POST_CHATGPT')['chatgpt_language_index'].apply(list)
ax.boxplot([pre_post_data[0], pre_post_data[1]], 
           labels=['Pre-ChatGPT\n(≤Nov 30, 2022)', 'Post-ChatGPT\n(>Nov 30, 2022)'])
ax.set_ylabel('ChatGPT Language Index', fontsize=12)
ax.set_title('Index Comparison: Pre vs Post ChatGPT Launch', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'pre_vs_post_chatgpt.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: pre_vs_post_chatgpt.png")
plt.close()

# Plot 4: Index vs Rating by quartile
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='rating', by='index_quartile', ax=ax)
ax.set_xlabel('ChatGPT Language Index Quartile', fontsize=12)
ax.set_ylabel('Average Rating', fontsize=12)
ax.set_title('Employee Ratings by ChatGPT Language Index Quartile', fontsize=14, fontweight='bold')
plt.suptitle('')  # Remove automatic title
plt.tight_layout()
plt.savefig(output_dir / 'index_vs_rating.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: index_vs_rating.png")
plt.close()

# Plot 5: Top 40 predictive words (combined)
top_20_combined = sig_weights.nlargest(20, 'coef')
bottom_20_combined = sig_weights.nsmallest(20, 'coef')
combined = pd.concat([top_20_combined, bottom_20_combined])

fig, ax = plt.subplots(figsize=(10, 14))
colors = ['green' if x > 0 else 'red' for x in combined['coef']]
y_labels = [f"{row['original_word']}" for _, row in combined.iterrows()]
ax.barh(range(len(combined)), combined['coef'], color=colors, alpha=0.7)
ax.set_yticks(range(len(combined)))
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel('Coefficient (Association with POST_CHATGPT)', fontsize=12)
ax.set_title('Top 40 Words: Language Changes After ChatGPT Launch', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(output_dir / 'top_predictive_words.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: top_predictive_words.png")
plt.close()

# ============================================================================
# 7. EXPORT FOR FURTHER ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("7. EXPORTING DATA")
print("="*70)

# Create analysis-ready dataset
analysis_df = df[[
    'ticker', 'year', 
    'chatgpt_language_index', 'chatgpt_index_pros', 'chatgpt_index_cons',
    'rating', 'review_count',
    'word_count_pros', 'word_count_cons',
    'POST_CHATGPT', 'index_quartile'
]].copy()

output_file = 'data/results/mnir/chatgpt_index_analysis_ready.csv'
analysis_df.to_csv(output_file, index=False)
print(f"\n✓ Saved analysis-ready dataset: {output_file}")
print(f"  Rows: {len(analysis_df):,}")
print(f"  Columns: {len(analysis_df.columns)}")

# Export top words to CSV for thesis
top_words_export = sig_weights.sort_values('coef', ascending=False)[
    ['original_word', 'word', 'section', 'coef', 't_stat', 'p_value']
].head(50)
top_words_export.to_csv('data/results/mnir/top_50_chatgpt_words.csv', index=False)
print(f"\n✓ Saved top 50 ChatGPT words: data/results/mnir/top_50_chatgpt_words.csv")

# Summary statistics for thesis
summary_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'N',
                  'Pre-ChatGPT Mean', 'Post-ChatGPT Mean', 'Difference', 'T-statistic', 'P-value'],
    'Value': [
        df['chatgpt_language_index'].mean(),
        df['chatgpt_language_index'].median(),
        df['chatgpt_language_index'].std(),
        df['chatgpt_language_index'].min(),
        df['chatgpt_language_index'].max(),
        len(df),
        pre_mean,
        post_mean,
        diff,
        t_stat,
        p_val
    ]
})
summary_stats.to_csv('data/results/mnir/index_summary_stats.csv', index=False)
print(f"✓ Saved summary statistics: data/results/mnir/index_summary_stats.csv")

# ============================================================================
# 8. INTERPRETATION GUIDE
# ============================================================================
print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)

print("\n📊 Generated Files:")
print("  1. chatgpt_index_analysis_ready.csv - Ready to merge with REIT performance data")
print("  2. top_50_chatgpt_words.csv - Top words for thesis")
print("  3. index_summary_stats.csv - Summary statistics table")
print("  4. figures/ - 5 visualization files")

print("\n📖 INTERPRETING THE INDEX:")
print("\n  The ChatGPT Language Index measures the extent to which firm-year")
print("  reviews use language patterns that emerged AFTER ChatGPT's launch")
print("  (November 30, 2022).")
print("\n  • HIGH index = Reviews use more 'post-ChatGPT' language")
print("  • LOW index = Reviews use more 'pre-ChatGPT' language")
print("\n  Positive coefficients = Words that INCREASED post-ChatGPT")
print("  Negative coefficients = Words that DECREASED post-ChatGPT")

print("\n🔍 KEY FINDINGS TO EXAMINE:")
print("\n  1. Did language patterns actually change after ChatGPT?")
print(f"     → {'YES' if p_val < 0.05 else 'NO'} (p={p_val:.4f})")
print(f"     → {abs(diff/pre_mean)*100:.1f}% change in average index")

print("\n  2. What words increased? (Check top_50_chatgpt_words.csv)")
print("     → Look for: technology terms, AI words, productivity language")

print("\n  3. What words decreased?")
print("     → Look for: traditional terms being replaced")

print("\n🚀 NEXT STEPS FOR YOUR THESIS:")
print("\n1. EXAMINE TOP WORDS:")
print("   Review top_50_chatgpt_words.csv to understand what language changed")
print("   Write about this in your 'Results' section")

print("\n2. MERGE WITH REIT PERFORMANCE DATA:")
print("   ```python")
print("   # Load your REIT performance data (returns, FFO, occupancy, etc.)")
print("   performance = pd.read_csv('your_reit_performance_data.csv')")
print("   ")
print("   # Merge on ticker and year")
print("   analysis_df = pd.read_csv('data/results/mnir/chatgpt_index_analysis_ready.csv')")
print("   merged = performance.merge(analysis_df, on=['ticker', 'year'], how='inner')")
print("   ```")

print("\n3. RUN REGRESSION ANALYSIS:")
print("   ```python")
print("   import statsmodels.formula.api as smf")
print("   ")
print("   # Does high ChatGPT language predict better performance?")
print("   model = smf.ols('returns ~ chatgpt_language_index + rating + review_count', ")
print("                   data=merged).fit()")
print("   print(model.summary())")
print("   ```")

print("\n4. DIFFERENCE-IN-DIFFERENCES:")
print("   ```python")
print("   # Did the effect of the index change post-ChatGPT?")
print("   model_did = smf.ols('''returns ~ chatgpt_language_index * POST_CHATGPT + ")
print("                          rating + review_count''', data=merged).fit()")
print("   print(model_did.summary())")
print("   ```")

print("\n5. HYPOTHESIS TESTS:")
print("   H1: High ChatGPT Language Index is associated with better firm performance")
print("   H2: Firms that adopted ChatGPT-era language faster performed better")
print("   H3: Effect varies by property type or firm size")

print("\n" + "="*70)