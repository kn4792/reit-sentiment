#!/usr/bin/env python3
"""
Results Visualization Script: Model Metrics Only

This script generates publication-quality plots from model results including just:
- Model metrics bar plot by split
- Metrics correlation heatmap

Usage:
    python scripts/generate_plots.py --input data/results/PLD/model_results --output-dir data/results/plots/PLD/

Author: Konain Niaz (kn4792@rit.edu)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import glob

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def get_latest_summary_csv(results_folder):
    """Return the path to the most recently created model_summary_*.csv file in a folder."""
    folder = Path(results_folder)
    csv_files = sorted(folder.glob("model_summary_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No model_summary_*.csv files found in {folder}")
    return str(csv_files[0])

def load_results(input_path):
    """
    Load model results from CSV file.
    """
    logger.info(f"Loading results from {input_path}")
    try:
        df = pd.read_csv(input_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        if 'split' in df.columns:
            logger.info(f"Splits: {df['split'].unique()}")
        return df
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise

def plot_model_performance_metrics(df, output_dir, out_prefix="model_performance_metrics"):
    """Generate model metrics plots: barplot by split and correlation heatmap."""

    # Barplot of metrics (works with csv in your example)
    logger.info("Generating model metrics barplot...")
    metrics = ['accuracy','precision_macro','recall_macro','f1_macro','r2','mse','rmse','mae']
    longform = df.melt(id_vars=[c for c in ['experiment', 'split'] if c in df.columns],
                       value_vars=[col for col in metrics if col in df.columns],
                       var_name="Metric", value_name="Value")
    # Filter to only rows with non-NaN and splits (train/Validation/Test)
    longform = longform[(~longform['Value'].isna()) & (longform['split'].notna())]

    # Only plot if any metrics found
    if not longform.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=longform, x="Metric", y="Value", hue="split")
        plt.title("Model Metrics by Split")
        plt.tight_layout()
        output_file = output_dir / f"{out_prefix}_bar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics barplot to {output_file}")
    else:
        logger.warning("No metrics available for barplot.")

    # Correlation heatmap: Use only numeric columns and enough rows
    logger.info("Generating correlation heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    corr_file = output_dir / f"{out_prefix}_corr.png"
    plt.savefig(corr_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved correlation heatmap to {corr_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate model metrics plots from results summary file (csv)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_plots.py --input data/results/PLD/model_results --output-dir data/results/plots/PLD/
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to results folder (containing model_summary_*.csv)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save plots')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Output format for plots (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for raster outputs (default: 300)')
    args = parser.parse_args()
    results_folder = args.input

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("RESULTS VISUALIZATION (model metrics only)")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Format: {args.format}")
    logger.info("="*80)

    try:
        summary_csv = get_latest_summary_csv(results_folder)
        df = load_results(summary_csv)
        if args.format != 'png':
            plt.rcParams['savefig.format'] = args.format

        plot_model_performance_metrics(df, output_dir)

        logger.info("="*80)
        logger.info("VISUALIZATION COMPLETE!")
        logger.info(f"All plots saved to: {output_dir}")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
