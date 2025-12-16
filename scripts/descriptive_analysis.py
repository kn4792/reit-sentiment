## @file descriptive_analysis.py
# @brief Stage 3: Descriptive Analysis and Visualization
#
# Creates comprehensive descriptive statistics and visualizations
# for the firm-year sentiment dataset.
#
# @details
# Input:  data/processed/firm_year_sentiment.csv
# Output: data/results/descriptive_stats/ (multiple files)
#
# Outputs generated:
# - Summary statistics tables (CSV)
# - Time trend visualizations (PNG)
# - Property type comparisons (PNG)
# - Pre/Post ChatGPT analysis (PNG)
# - Correlation matrices (CSV, PNG)
# - Analysis summary report (TXT)
#
# @author Konain Niaz (kn4792@rit.edu)
# @date 2025-12-16
# @version 1.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy import stats

print("=" * 80)
print("STAGE 3: DESCRIPTIVE ANALYSIS AND VISUALIZATION")
print("=" * 80)
print()

## @var INPUT_FILE
# Path to input CSV file with firm-year aggregated data
INPUT_FILE = "data/processed/firm_year_sentiment.csv"

## @var OUTPUT_DIR
# Directory for descriptive statistics output files
OUTPUT_DIR = "data/results/descriptive_stats"

## @var FIGURES_DIR
# Directory for visualization output files
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set visualization style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# STEP 1: LOAD DATA
# Load the firm-year aggregated dataset from Stage 2
# ============================================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} firm-year observations")
print(f"Columns: {len(df.columns)}")
print()

# ============================================================================
# STEP 2: DESCRIPTIVE STATISTICS TABLES
# Create summary statistics tables for key variables
# ============================================================================
print("Creating descriptive statistics tables...")

# Overall summary statistics for key variables
desc_stats = df[[
    'sentiment_score_mean', 'sentiment_dispersion', 'sentiment_agreement',
    'rating_mean', 'review_count', 'positive_prob_mean', 'negative_prob_mean'
]].describe()

desc_stats.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
print("Saved: summary_statistics.csv")

# Statistics grouped by property type
prop_stats = df.groupby('property_type')[[
    'sentiment_score_mean', 'rating_mean', 'review_count'
]].agg(['mean', 'std', 'count'])

prop_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_by_property_type.csv"))
print("Saved: stats_by_property_type.csv")

# Statistics grouped by Pre/Post ChatGPT
chatgpt_stats = df.groupby('POST_CHATGPT')[[
    'sentiment_score_mean', 'sentiment_dispersion', 'rating_mean', 'review_count'
]].agg(['mean', 'std', 'count'])

chatgpt_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_pre_post_chatgpt.csv"))
print("Saved: stats_pre_post_chatgpt.csv")

# Statistics grouped by year
year_stats = df.groupby('year')[[
    'sentiment_score_mean', 'rating_mean', 'review_count'
]].agg(['mean', 'std', 'count'])

year_stats.to_csv(os.path.join(OUTPUT_DIR, "stats_by_year.csv"))
print("Saved: stats_by_year.csv")

print()

# ============================================================================
# STEP 3: CORRELATION MATRIX
# Calculate and visualize correlations between key variables
# ============================================================================
print("Creating correlation matrix...")

## @var corr_vars
# List of variables to include in correlation analysis
corr_vars = [
    'sentiment_score_mean', 'sentiment_dispersion', 'sentiment_agreement',
    'rating_mean', 'review_count', 'POST_CHATGPT', 'tech_intensive'
]

# Calculate Pearson correlation matrix
corr_matrix = df[corr_vars].corr()
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
print("Saved: correlation_matrix.csv")

# Create correlation heatmap visualization
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Key Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")
print()

# ============================================================================
# STEP 4: TIME TRENDS
# Visualize sentiment and rating trends over time
# ============================================================================
print("Creating time trend visualizations...")

# Calculate yearly aggregates with confidence intervals
yearly = df.groupby('year').agg({
    'sentiment_score_mean': ['mean', 'std', 'count'],
    'rating_mean': 'mean'
}).reset_index()

yearly.columns = ['year', 'sentiment_mean', 'sentiment_std', 'n_firms', 'rating_mean']
yearly['sentiment_se'] = yearly['sentiment_std'] / np.sqrt(yearly['n_firms'])

# Create dual panel figure for sentiment and rating trends
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Sentiment trend with confidence interval
ax1.plot(yearly['year'], yearly['sentiment_mean'], marker='o', linewidth=2, 
         markersize=8, color='steelblue', label='Mean Sentiment')
ax1.fill_between(yearly['year'],
                 yearly['sentiment_mean'] - 1.96 * yearly['sentiment_se'],
                 yearly['sentiment_mean'] + 1.96 * yearly['sentiment_se'],
                 alpha=0.3, color='steelblue', label='95% CI')
# Add vertical line for ChatGPT launch (Nov 30, 2022 ≈ 2022.917)
ax1.axvline(x=2022.917, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ChatGPT Launch')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Sentiment Score', fontsize=12)
ax1.set_title('Employee Sentiment Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Rating trend
ax2.plot(yearly['year'], yearly['rating_mean'], marker='s', linewidth=2,
         markersize=8, color='coral', label='Mean Rating')
ax2.axvline(x=2022.917, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ChatGPT Launch')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Glassdoor Rating (1-5)', fontsize=12)
ax2.set_title('Employee Ratings Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "time_trends.png"), bbox_inches='tight')
plt.close()
print("Saved: time_trends.png")

# Create review volume bar chart
fig, ax = plt.subplots(figsize=(12, 6))
yearly_reviews = df.groupby('year')['review_count'].sum()
ax.bar(yearly_reviews.index, yearly_reviews.values, color='forestgreen', alpha=0.7)
ax.axvline(x=2022.917, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ChatGPT Launch')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Total Reviews', fontsize=12)
ax.set_title('Review Volume Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "review_volume_trend.png"), bbox_inches='tight')
plt.close()
print("Saved: review_volume_trend.png")
print()

# ============================================================================
# STEP 5: PRE/POST CHATGPT COMPARISON
# Create box plots comparing metrics before and after ChatGPT launch
# ============================================================================
print("Creating Pre/Post ChatGPT comparisons...")

# Create 2x2 grid of box plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

## @var metrics
# List of tuples defining (variable, label, axis) for box plots
metrics = [
    ('sentiment_score_mean', 'Sentiment Score', axes[0, 0]),
    ('sentiment_dispersion', 'Sentiment Dispersion', axes[0, 1]),
    ('rating_mean', 'Glassdoor Rating', axes[1, 0]),
    ('review_count', 'Reviews per Firm-Year', axes[1, 1])
]

# Create box plot for each metric with t-test results
for var, label, ax in metrics:
    pre_data = df[df['POST_CHATGPT'] == 0][var]
    post_data = df[df['POST_CHATGPT'] == 1][var]
    
    bp = ax.boxplot([pre_data, post_data], labels=['Pre-ChatGPT', 'Post-ChatGPT'],
                     patch_artist=True, widths=0.6)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    # Add means as diamond markers
    ax.plot([1, 2], [pre_data.mean(), post_data.mean()], 
            'D-', color='darkred', markersize=10, linewidth=2, label='Mean')
    
    # Perform t-test and display results
    t_stat, p_val = stats.ttest_ind(pre_data, post_data)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.text(0.5, 0.95, f't={t_stat:.2f}, p={p_val:.4f} {sig_text}',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label}: Pre vs Post ChatGPT', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pre_post_chatgpt_comparison.png"), bbox_inches='tight')
plt.close()
print("Saved: pre_post_chatgpt_comparison.png")
print()

# ============================================================================
# STEP 6: PROPERTY TYPE ANALYSIS
# Compare sentiment across different REIT property types
# ============================================================================
print("Creating property type comparisons...")

# Calculate mean sentiment by property type
prop_summary = df.groupby('property_type').agg({
    'sentiment_score_mean': ['mean', 'std', 'count'],
    'rating_mean': 'mean',
    'tech_intensive': 'first'
}).reset_index()

prop_summary.columns = ['property_type', 'sentiment_mean', 'sentiment_std', 'n_obs', 'rating', 'tech_intensive']
prop_summary = prop_summary.sort_values('sentiment_mean', ascending=False)

# Create horizontal bar chart for top 12 property types
prop_summary_top = prop_summary.head(12)

fig, ax = plt.subplots(figsize=(12, 8))
# Color bars: green for tech-intensive, blue for traditional
colors = ['darkgreen' if x == 1 else 'steelblue' for x in prop_summary_top['tech_intensive']]
bars = ax.barh(range(len(prop_summary_top)), prop_summary_top['sentiment_mean'], color=colors, alpha=0.7)

# Add error bars for standard deviation
ax.errorbar(prop_summary_top['sentiment_mean'], range(len(prop_summary_top)),
            xerr=prop_summary_top['sentiment_std'], fmt='none', ecolor='black', alpha=0.5)

ax.set_yticks(range(len(prop_summary_top)))
ax.set_yticklabels(prop_summary_top['property_type'], fontsize=10)
ax.set_xlabel('Mean Sentiment Score', fontsize=12)
ax.set_title('Employee Sentiment by Property Type\n(Green = Technology-Intensive REITs)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Add sample sizes
for i, n in enumerate(prop_summary_top['n_obs']):
    ax.text(ax.get_xlim()[1] * 0.95, i, f'n={n}', va='center', ha='right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "sentiment_by_property_type.png"), bbox_inches='tight')
plt.close()
print("  ✓ Saved: sentiment_by_property_type.png")

# Create tech-intensive vs traditional comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Box plot comparison
tech_data = df[df['tech_intensive'] == 1]['sentiment_score_mean']
trad_data = df[df['tech_intensive'] == 0]['sentiment_score_mean']

bp1 = axes[0].boxplot([tech_data, trad_data], 
                       labels=['Tech-Intensive', 'Traditional'],
                       patch_artist=True, widths=0.6)
bp1['boxes'][0].set_facecolor('darkgreen')
bp1['boxes'][1].set_facecolor('steelblue')

axes[0].plot([1, 2], [tech_data.mean(), trad_data.mean()],
             'D-', color='darkred', markersize=10, linewidth=2, label='Mean')

t_stat, p_val = stats.ttest_ind(tech_data, trad_data)
sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
axes[0].text(0.5, 0.95, f't={t_stat:.2f}, p={p_val:.4f} {sig_text}',
            transform=axes[0].transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_ylabel('Sentiment Score', fontsize=11)
axes[0].set_title('Sentiment: Tech-Intensive vs Traditional REITs', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Panel 2: Time trend by REIT type
yearly_by_type = df.groupby(['year', 'tech_intensive'])['sentiment_score_mean'].mean().reset_index()
for tech_val, label, color in [(1, 'Tech-Intensive', 'darkgreen'), (0, 'Traditional', 'steelblue')]:
    data = yearly_by_type[yearly_by_type['tech_intensive'] == tech_val]
    axes[1].plot(data['year'], data['sentiment_score_mean'], 
                marker='o', linewidth=2, markersize=6, label=label, color=color)

axes[1].axvline(x=2022.917, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ChatGPT')
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Mean Sentiment Score', fontsize=11)
axes[1].set_title('Sentiment Trends by REIT Type', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tech_vs_traditional_comparison.png"), bbox_inches='tight')
plt.close()
print("Saved: tech_vs_traditional_comparison.png")
print()

# ============================================================================
# STEP 7: SENTIMENT DISTRIBUTION
# Visualize the distribution of sentiment scores
# ============================================================================
print("Creating sentiment distribution plots...")

# Create 3x2 grid: overall + top 5 property types
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# Panel 1: Overall distribution
axes[0, 0].hist(df['sentiment_score_mean'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(df['sentiment_score_mean'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {df["sentiment_score_mean"].mean():.3f}')
axes[0, 0].axvline(df['sentiment_score_mean'].median(), color='orange', linestyle='--',
                   linewidth=2, label=f'Median = {df["sentiment_score_mean"].median():.3f}')
axes[0, 0].set_xlabel('Sentiment Score', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Distribution of Firm-Year Sentiment', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Panels 2-6: Top 5 property types
top_types = df['property_type'].value_counts().head(5).index
positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

for i, (ptype, (row, col)) in enumerate(zip(top_types, positions)):
    data = df[df['property_type'] == ptype]['sentiment_score_mean']
    axes[row, col].hist(data, bins=30, color=f'C{i}', alpha=0.7, edgecolor='black')
    axes[row, col].axvline(data.mean(), color='red', linestyle='--', linewidth=2)
    axes[row, col].set_xlabel('Sentiment Score', fontsize=10)
    axes[row, col].set_ylabel('Frequency', fontsize=10)
    axes[row, col].set_title(f'{ptype} (n={len(data)})', fontsize=11, fontweight='bold')
    axes[row, col].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "sentiment_distributions.png"), bbox_inches='tight')
plt.close()
print("Saved: sentiment_distributions.png")
print()

# ============================================================================
# STEP 8: CREATE SUMMARY REPORT
# Generate text report summarizing all analyses
# ============================================================================
print("Creating summary report...")

report = f"""
DESCRIPTIVE ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

DATASET OVERVIEW
- Total firm-year observations: {len(df):,}
- Unique firms: {df['ticker'].nunique()}
- Year range: {df['year'].min()} - {df['year'].max()}
- Average reviews per firm-year: {df['review_count'].mean():.1f}

SENTIMENT STATISTICS
- Mean sentiment score: {df['sentiment_score_mean'].mean():.4f}
- Median sentiment score: {df['sentiment_score_mean'].median():.4f}
- Std deviation: {df['sentiment_score_mean'].std():.4f}
- Range: [{df['sentiment_score_mean'].min():.4f}, {df['sentiment_score_mean'].max():.4f}]

PRE/POST CHATGPT COMPARISON
- Pre-ChatGPT firm-years: {(df['POST_CHATGPT']==0).sum():,}
- Post-ChatGPT firm-years: {(df['POST_CHATGPT']==1).sum():,}
- Pre-ChatGPT sentiment: {df[df['POST_CHATGPT']==0]['sentiment_score_mean'].mean():.4f}
- Post-ChatGPT sentiment: {df[df['POST_CHATGPT']==1]['sentiment_score_mean'].mean():.4f}
- Difference: {df[df['POST_CHATGPT']==1]['sentiment_score_mean'].mean() - df[df['POST_CHATGPT']==0]['sentiment_score_mean'].mean():.4f}

PROPERTY TYPE ANALYSIS
- Tech-intensive REITs: {(df['tech_intensive']==1).sum():,} firm-years
- Traditional REITs: {(df['tech_intensive']==0).sum():,} firm-years
- Tech-intensive sentiment: {df[df['tech_intensive']==1]['sentiment_score_mean'].mean():.4f}
- Traditional sentiment: {df[df['tech_intensive']==0]['sentiment_score_mean'].mean():.4f}

TOP 5 PROPERTY TYPES BY SENTIMENT
{prop_summary.head()[['property_type', 'sentiment_mean', 'n_obs']].to_string(index=False)}

FILES GENERATED
- Summary statistics tables (CSV)
- Correlation matrix
- Time trend visualizations
- Pre/Post ChatGPT comparisons
- Property type analysis
- Sentiment distributions

================================================================================
"""

report_file = os.path.join(OUTPUT_DIR, "analysis_summary_report.txt")
with open(report_file, 'w') as f:
    f.write(report)
print("Saved: analysis_summary_report.txt")
print()

print("=" * 80)
print("STAGE 3 COMPLETE")
print("=" * 80)
print()
print("Output Directory: " + OUTPUT_DIR)
print()
print("Generated Files:")
print("Tables:")
print("summary_statistics.csv")
print("stats_by_property_type.csv")
print("stats_pre_post_chatgpt.csv")
print("stats_by_year.csv")
print("correlation_matrix.csv")
print()
print("Figures:")
print("correlation_heatmap.png")
print("time_trends.png")
print("review_volume_trend.png")
print("pre_post_chatgpt_comparison.png")
print("sentiment_by_property_type.png")
print("tech_vs_traditional_comparison.png")
print("sentiment_distributions.png")
print()
print("Reports:")
print("analysis_summary_report.txt")
print()
print("Next Step:")
print("Run Stage 4: python scripts/export_analysis_dataset.py")
print("=" * 80)