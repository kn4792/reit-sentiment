#!/usr/bin/env python3
"""
Visualization & Plotting Script

Generates comprehensive visualizations for REIT sentiment analysis results.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    python scripts/generate_plots.py --input data/results/model_results.csv \\
                                     --output-dir data/results/plots/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_sentiment_distribution(df, output_dir):
    """Plot sentiment score distribution."""
    if 'sentiment_compound' not in df.columns:
        print("  ⚠️ No sentiment_compound column, skipping")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['sentiment_compound'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5, label='Neutral')
    axes[0].set_xlabel('Sentiment Compound Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sentiment Scores')
    axes[0].legend()
    
    # Box plot by company (top 10)
    if 'ticker' in df.columns:
        top_companies = df['ticker'].value_counts().head(10).index
        df_top = df[df['ticker'].isin(top_companies)]
        df_top.boxplot(column='sentiment_compound', by='ticker', ax=axes[1])
        axes[1].set_xlabel('Company Ticker')
        axes[1].set_ylabel('Sentiment Compound Score')
        axes[1].set_title('Sentiment by Top 10 Companies')
        plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ sentiment_distribution.png")


def plot_sentiment_over_time(df, output_dir):
    """Plot sentiment trends over time."""
    if 'date' not in df.columns or 'sentiment_compound' not in df.columns:
        print("  ⚠️ Missing required columns, skipping time series")
        return
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Monthly aggregation
    monthly = df.groupby('year_month')['sentiment_compound'].agg(['mean', 'std', 'count'])
    monthly = monthly[monthly['count'] >= 5]  # Filter months with few reviews
    
    if len(monthly) == 0:
        print("  ⚠️ Insufficient data for time series")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(monthly))
    ax.plot(x, monthly['mean'], marker='o', linewidth=2, label='Mean Sentiment')
    ax.fill_between(x,
                    monthly['mean'] - monthly['std'],
                    monthly['mean'] + monthly['std'],
                    alpha=0.2, label='±1 Std Dev')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Sentiment')
    ax.set_title('Sentiment Trend Over Time')
    ax.set_xticks(x[::max(1, len(x)//12)])
    ax.set_xticklabels(monthly.index.astype(str)[::max(1, len(x)//12)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sentiment_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ sentiment_time_series.png")


def plot_company_comparison(df, output_dir):
    """Plot company-level comparisons."""
    if 'ticker' not in df.columns or 'sentiment_compound' not in df.columns:
        print("  ⚠️ Missing required columns, skipping company comparison")
        return
    
    # Top 15 companies by review count
    top_companies = df['ticker'].value_counts().head(15).index
    df_top = df[df['ticker'].isin(top_companies)]
    
    company_stats = df_top.groupby('ticker')['sentiment_compound'].agg(['mean', 'count'])
    company_stats = company_stats.sort_values('mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['red' if x < 0 else 'green' for x in company_stats['mean']]
    ax.barh(range(len(company_stats)), company_stats['mean'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(company_stats)))
    ax.set_yticklabels(company_stats.index)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Average Sentiment Score')
    ax.set_ylabel('Company Ticker')
    ax.set_title('Average Sentiment by Company (Top 15)')
    
    # Add review counts as annotations
    for i, (idx, row) in enumerate(company_stats.iterrows()):
        ax.text(row['mean'], i, f" n={int(row['count'])}", va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'company_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ company_comparison.png")


def plot_ai_adoption(df, output_dir):
    """Plot AI adoption trends."""
    if 'ai_mentioned' not in df.columns or 'date' not in df.columns:
        print("  ⚠️ Missing AI columns, skipping AI adoption plot")
        return
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Monthly AI mentions
    monthly_ai = df.groupby('year_month').agg({
        'ai_mentioned': 'sum',
        'ticker': 'count'
    })
    monthly_ai['ai_pct'] = (monthly_ai['ai_mentioned'] / monthly_ai['ticker']) * 100
    
    if len(monthly_ai) == 0:
        print("  ⚠️ No AI data available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(monthly_ai))
    ax.plot(x, monthly_ai['ai_pct'], marker='o', linewidth=2, color='purple')
    ax.fill_between(x, 0, monthly_ai['ai_pct'], alpha=0.2, color='purple')
    
    ax.set_xlabel('Month')
    ax.set_ylabel('AI Mention Percentage (%)')
    ax.set_title('AI/Technology Adoption Trend Over Time')
    ax.set_xticks(x[::max(1, len(x)//12)])
    ax.set_xticklabels(monthly_ai.index.astype(str)[::max(1, len(x)//12)], rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ai_adoption_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ ai_adoption_trend.png")


def plot_model_results(results_file, output_dir):
    """Plot model performance results."""
    if not results_file.exists():
        print("  ⚠️ Model results file not found, skipping")
        return
    
    # Try to load JSON results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except:
        print("  ⚠️ Could not load model results JSON")
        return
    
    # Extract metrics
    exp_data = []
    for exp_name, exp_results in results.items():
        for split, metrics in exp_results.items():
            if isinstance(metrics, dict):
                exp_data.append({
                    'experiment': exp_name,
                    'split': split,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                })
    
    if not exp_data:
        print("  ⚠️ No model metrics to plot")
        return
    
    df_metrics = pd.DataFrame(exp_data)
    
    # Plot classification metrics
    if 'accuracy' in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        splits = ['train', 'val', 'test']
        metrics_to_plot = ['accuracy', 'f1_macro']
        
        x = np.arange(len(splits))
        width = 0.35
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df_metrics.columns:
                values = [df_metrics[(df_metrics['split'] == s) & (df_metrics['experiment'] == 'classification')][metric].values[0]
                         if len(df_metrics[(df_metrics['split'] == s) & (df_metrics['experiment'] == 'classification')]) > 0
                         else 0 for s in splits]
                ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Dataset Split')
        ax.set_ylabel('Score')
        ax.set_title('Classification Model Performance')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([s.capitalize() for s in splits])
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_classification_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ model_classification_performance.png")
    
    # Plot regression metrics
    if 'r2' in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        splits = ['train', 'val', 'test']
        
        r2_values = [df_metrics[(df_metrics['split'] == s) & (df_metrics['experiment'] == 'regression')]['r2'].values[0]
                     if len(df_metrics[(df_metrics['split'] == s) & (df_metrics['experiment'] == 'regression')]) > 0
                     else 0 for s in splits]
        
        ax.bar(splits, r2_values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Dataset Split')
        ax.set_ylabel('R² Score')
        ax.set_title('Regression Model Performance')
        ax.set_ylim([0, 1])
        
        for i, v in enumerate(r2_values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_regression_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ model_regression_performance.png")


def generate_all_plots(input_file, output_dir, results_file=None):
    """Generate all plots."""
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print('='*60)
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} records")
    
    # Generate plots
    print("\n📊 Generating plots...")
    
    plot_sentiment_distribution(df, output_dir)
    plot_sentiment_over_time(df, output_dir)
    plot_company_comparison(df, output_dir)
    plot_ai_adoption(df, output_dir)
    
    # Plot model results if provided
    if results_file:
        plot_model_results(Path(results_file), output_dir)
    
    print(f"\n{'='*60}")
    print("✅ VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for REIT sentiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to sentiment data CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/results/plots',
        help='Output directory for plots (default: data/results/plots)'
    )
    parser.add_argument(
        '--results',
        type=str,
        help='Optional: Path to model results JSON for performance plots'
    )
    
    args = parser.parse_args()
    
    # Validate input exists
    if not Path(args.input).exists():
        print(f"✗ Error: Input file not found: {args.input}")
        return
    
    # Generate plots
    try:
        generate_all_plots(
            input_file=args.input,
            output_dir=args.output_dir,
            results_file=args.results
        )
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        raise


if __name__ == '__main__':
    main()