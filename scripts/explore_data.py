#!/usr/bin/env python3

"""
Exploratory Data Analysis Script

This script performs exploratory data analysis on cleaned REIT review data,
generating distribution plots, time-series visualizations, summary tables, and
integrates FinBERT sentiment scoring for text columns.

Usage:
    python scripts/explore_data.py --input data/processed/cleaned_all_reviews.csv --output-dir data/results/exploration/
    python scripts/explore_data.py --input data/processed/cleaned_PLD_reviews.csv --output-dir data/results/exploration/PLD/

Author: Konain Niaz (kn4792@rit.edu)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import sentiment analyzer
from scripts.finbert_validation import FinBERTSentimentAnalyzer

from src.visualization.plotting import (
    generate_distribution_plots,
    generate_timeseries_plots
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data(input_path):
    """
    Load data from CSV file.
    Args:
        input_path (str): Path to input CSV file
    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        if 'date' in df.columns:
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def add_sentiment_scores(df, text_col='pros_cleaned', batch_size=16, use_gpu=False):
    """
    Compute FinBERT sentiment scores for each review and add as a column.
    Args:
        df (pd.DataFrame): DataFrame with reviews
        text_col (str): Column to analyze
    Returns:
        pd.DataFrame: DataFrame with sentiment_score column
    """
    analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=use_gpu, batch_size=batch_size)
    logger.info(f"Computing sentiment scores from '{text_col}' using FinBERT...")
    df['sentiment_score'] = df[text_col].astype(str).apply(lambda text: analyzer.predict_single(text)['compound'])
    logger.info("Done computing sentiment scores!")
    return df

def generate_distribution_plots(df, output_dir):
    """
    Generate distribution plots for ratings and sentiment scores.
    Args:
        df (pd.DataFrame): Input data
        output_dir (Path): Output directory
    """
    logger.info("Generating distribution plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Rating distribution
    if 'rating' in df.columns:
        axes[0].hist(df['rating'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Rating', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Ratings', fontsize=14, fontweight='bold')
        axes[0].axvline(df['rating'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f"Mean: {df['rating'].mean():.2f}")

        axes[0].legend()

    # Sentiment distribution
    if 'sentiment_score' in df.columns:
        axes[1].hist(df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
        axes[1].set_xlabel('Sentiment Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Sentiment Scores', fontsize=14, fontweight='bold')
        axes[1].legend()

    plt.tight_layout()
    output_file = output_dir / "distribution_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plots to {output_file}")

def generate_timeseries_plots(df, output_dir):
    """
    Generate time-series plots for sentiment and ratings over time.
    Args:
        df (pd.DataFrame): Input data
        output_dir (Path): Output directory
    """
    logger.info("Generating time-series plots...")
    if 'date' not in df.columns:
        logger.warning("No 'date' column found. Skipping time-series plots.")
        return
    df = df.sort_values('date')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    if 'sentiment_score' in df.columns:
        monthly_sentiment = df.groupby(pd.Grouper(key='date', freq='M'))['sentiment_score'].mean()
        axes[0].plot(df['date'], df['sentiment_score'], alpha=0.3, color='lightblue', label='Individual Reviews')
        axes[0].plot(monthly_sentiment.index, monthly_sentiment.values, color='darkblue', linewidth=2, label='Monthly Average')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Sentiment Score', fontsize=12)
        axes[0].set_title('Sentiment Over Time', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    if 'rating' in df.columns:
        monthly_rating = df.groupby(pd.Grouper(key='date', freq='M'))['rating'].mean()
        axes[1].plot(df['date'], df['rating'], alpha=0.3, color='lightcoral', label='Individual Reviews')
        axes[1].plot(monthly_rating.index, monthly_rating.values, color='darkred', linewidth=2, label='Monthly Average')
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Rating', fontsize=12)
        axes[1].set_title('Rating Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_file = output_dir / "timeseries_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved time-series plots to {output_file}")

def generate_statistical_summary(df, output_dir):
    """
    Generate statistical summary tables.
    Args:
        df (pd.DataFrame): Input data
        output_dir (Path): Output directory
    """
    logger.info("Generating statistical summary...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'index']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe()
        summary_file = output_dir / "summary_statistics.csv"
        summary.to_csv(summary_file)
        logger.info(f"Saved summary statistics to {summary_file}")

        # Print summary to console for quick review
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary.to_string())
        print("="*80 + "\n")

    # Monthly summary (if date column exists)
    if 'date' in df.columns:
        monthly_summary = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            col: ['mean', 'std', 'count']
            for col in numeric_cols if col in df.columns
        })
        monthly_file = output_dir / "monthly_summary.csv"
        monthly_summary.to_csv(monthly_file)
        logger.info(f"Saved monthly summary to {monthly_file}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Perform exploratory data analysis on REIT review data (now with FinBERT sentiment integration)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/explore_data.py --input data/processed/cleaned_all_reviews.csv --output-dir data/results/exploration/
  python scripts/explore_data.py --input data/processed/cleaned_PLD_reviews.csv --output-dir data/results/exploration/PLD/
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save exploration outputs')
    parser.add_argument('--text-col', type=str, default='pros_cleaned',
                        help='Column to use for sentiment scoring (default: pros_cleaned)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for FinBERT sentiment analysis')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for FinBERT inference')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("EXPLORATORY DATA ANALYSIS (with FinBERT Sentiment)")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*80)

    try:
        # Load data
        df = load_data(args.input)

        # Add sentiment scores if not already present
        if 'sentiment_score' not in df.columns:
            df = add_sentiment_scores(df, text_col=args.text_col, batch_size=args.batch_size, use_gpu=args.use_gpu)

        # Generate distribution and time-series plots
        generate_distribution_plots(df, output_dir)
        generate_timeseries_plots(df, output_dir)
        generate_statistical_summary(df, output_dir)

        logger.info("="*80)
        logger.info("EXPLORATION COMPLETE!")
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Error during exploration: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
