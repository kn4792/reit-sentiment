#!/usr/bin/env python3
"""
Results Table Generation Script

This script generates publication-quality tables from model results including:
- Summary statistics tables
- Company comparison tables
- Time period analysis tables
- LaTeX formatted tables for thesis

Usage:
    python scripts/generate_tables.py --input data/results/model_results.csv --output-dir data/results/tables/
    python scripts/generate_tables.py --input data/results/model_results.csv --format latex
    python scripts/generate_tables.py --input data/results/model_results.csv --table-types summary,company

Author: Konain Niaz (kn4792@rit.edu)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def load_results(input_path):
    """
    Load model results from CSV file.
    
    Args:
        input_path (str): Path to input CSV file
        
    Returns:
        pd.DataFrame: Loaded results
    """
    logger.info(f"Loading results from {input_path}")
    
    try:
        df = pd.read_csv(input_path)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise


def generate_summary_statistics_table(df, output_dir, format='csv'):
    """
    Generate overall summary statistics table.
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
    """
    logger.info("Generating summary statistics table...")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'index']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found for summary statistics")
        return
    
    # Generate summary statistics
    summary = df[numeric_cols].describe().T
    summary['median'] = df[numeric_cols].median()
    
    # Reorder columns
    col_order = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
    summary = summary[[col for col in col_order if col in summary.columns]]
    
    # Round values
    summary = summary.round(3)
    
    # Save to CSV
    csv_file = output_dir / "summary_statistics.csv"
    summary.to_csv(csv_file)
    logger.info(f"Saved summary statistics to {csv_file}")
    
    # Save to LaTeX if requested
    if format == 'latex':
        latex_str = summary.to_latex(
            caption='Summary Statistics of Model Results',
            label='tab:summary_stats',
            column_format='l' + 'r' * len(summary.columns),
            float_format='%.3f'
        )
        
        latex_file = output_dir / "summary_statistics.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary.to_string())
    print("="*80 + "\n")


def generate_company_comparison_table(df, output_dir, format='csv'):
    """
    Generate company-level comparison table.
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
    """
    logger.info("Generating company comparison table...")
    
    if 'company' not in df.columns:
        logger.warning("No 'company' column found")
        return
    
    # Define aggregation metrics
    agg_dict = {}
    
    if 'sentiment_score' in df.columns:
        agg_dict['sentiment_score'] = ['mean', 'std']
    
    if 'rating' in df.columns:
        agg_dict['rating'] = ['mean', 'std']
    
    if 'ai_mentions' in df.columns:
        agg_dict['ai_mentions'] = 'sum'
    
    if 'review_count' in df.columns:
        agg_dict['review_count'] = 'sum'
    else:
        # Count rows per company
        agg_dict['company'] = 'count'
    
    if 'revenue_growth' in df.columns:
        agg_dict['revenue_growth'] = 'mean'
    
    if 'employee_growth' in df.columns:
        agg_dict['employee_growth'] = 'mean'
    
    # Generate comparison table
    company_summary = df.groupby('company').agg(agg_dict)
    
    # Flatten column names
    company_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                               for col in company_summary.columns.values]
    
    # Round values
    company_summary = company_summary.round(3)
    
    # Sort by sentiment score (if available)
    if 'sentiment_score_mean' in company_summary.columns:
        company_summary = company_summary.sort_values('sentiment_score_mean', ascending=False)
    
    # Save to CSV
    csv_file = output_dir / "company_comparison.csv"
    company_summary.to_csv(csv_file)
    logger.info(f"Saved company comparison to {csv_file}")
    
    # Save to LaTeX if requested
    if format == 'latex':
        latex_str = company_summary.to_latex(
            caption='Company-Level Comparison of Key Metrics',
            label='tab:company_comparison',
            column_format='l' + 'r' * len(company_summary.columns),
            float_format='%.3f'
        )
        
        latex_file = output_dir / "company_comparison.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("COMPANY COMPARISON")
    print("="*80)
    print(company_summary.to_string())
    print("="*80 + "\n")


def generate_time_period_comparison_table(df, output_dir, format='csv'):
    """
    Generate time period comparison table (before/after ChatGPT).
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
    """
    logger.info("Generating time period comparison table...")
    
    if 'date' not in df.columns:
        logger.warning("No 'date' column found")
        return
    
    # Define ChatGPT launch date
    chatgpt_date = pd.Timestamp('2022-11-30')
    
    # Split data
    before = df[df['date'] < chatgpt_date]
    after = df[df['date'] >= chatgpt_date]
    
    # Define metrics to compare
    metrics = []
    if 'sentiment_score' in df.columns:
        metrics.append('sentiment_score')
    if 'rating' in df.columns:
        metrics.append('rating')
    if 'ai_mentions' in df.columns:
        metrics.append('ai_mentions')
    
    # Create comparison table
    comparison_data = []
    
    for metric in metrics:
        before_mean = before[metric].mean()
        after_mean = after[metric].mean()
        
        before_std = before[metric].std()
        after_std = after[metric].std()
        
        # Calculate change
        abs_change = after_mean - before_mean
        pct_change = (abs_change / abs(before_mean) * 100) if before_mean != 0 else 0
        
        comparison_data.append({
            'Metric': metric,
            'Before_Mean': before_mean,
            'Before_Std': before_std,
            'Before_N': len(before),
            'After_Mean': after_mean,
            'After_Std': after_std,
            'After_N': len(after),
            'Abs_Change': abs_change,
            'Pct_Change': pct_change
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(3)
    
    # Save to CSV
    csv_file = output_dir / "time_period_comparison.csv"
    comparison_df.to_csv(csv_file, index=False)
    logger.info(f"Saved time period comparison to {csv_file}")
    
    # Save to LaTeX if requested
    if format == 'latex':
        latex_str = comparison_df.to_latex(
            caption='Comparison of Metrics Before and After ChatGPT Launch (Nov 30, 2022)',
            label='tab:time_comparison',
            index=False,
            column_format='l' + 'r' * (len(comparison_df.columns) - 1),
            float_format='%.3f'
        )
        
        latex_file = output_dir / "time_period_comparison.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("TIME PERIOD COMPARISON (Before vs After ChatGPT Launch)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")


def generate_correlation_table(df, output_dir, format='csv'):
    """
    Generate correlation matrix table.
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
    """
    logger.info("Generating correlation table...")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'index']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric columns for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    corr_matrix = corr_matrix.round(3)
    
    # Save to CSV
    csv_file = output_dir / "correlation_matrix.csv"
    corr_matrix.to_csv(csv_file)
    logger.info(f"Saved correlation matrix to {csv_file}")
    
    # Save to LaTeX if requested
    if format == 'latex':
        latex_str = corr_matrix.to_latex(
            caption='Correlation Matrix of Key Variables',
            label='tab:correlation',
            column_format='l' + 'r' * len(corr_matrix.columns),
            float_format='%.3f'
        )
        
        latex_file = output_dir / "correlation_matrix.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Print to console
    print("\n" + "="*80)
    print("CORRELATION MATRIX")
    print("="*80)
    print(corr_matrix.to_string())
    print("="*80 + "\n")


def generate_monthly_trends_table(df, output_dir, format='csv'):
    """
    Generate monthly aggregated trends table.
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
    """
    logger.info("Generating monthly trends table...")
    
    if 'date' not in df.columns:
        logger.warning("No 'date' column found")
        return
    
    # Define aggregation metrics
    agg_dict = {}
    
    if 'sentiment_score' in df.columns:
        agg_dict['sentiment_score'] = ['mean', 'std', 'count']
    
    if 'rating' in df.columns:
        agg_dict['rating'] = ['mean', 'std']
    
    if 'ai_mentions' in df.columns:
        agg_dict['ai_mentions'] = 'sum'
    
    # Aggregate by month
    monthly = df.groupby(pd.Grouper(key='date', freq='M')).agg(agg_dict)
    
    # Flatten column names
    monthly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in monthly.columns.values]
    
    # Round values
    monthly = monthly.round(3)
    
    # Reset index to make date a column
    monthly = monthly.reset_index()
    
    # Format date as string
    monthly['date'] = monthly['date'].dt.strftime('%Y-%m')
    
    # Save to CSV
    csv_file = output_dir / "monthly_trends.csv"
    monthly.to_csv(csv_file, index=False)
    logger.info(f"Saved monthly trends to {csv_file}")
    
    # Save to LaTeX if requested (only recent months to avoid too long table)
    if format == 'latex':
        # Take last 12 months only for LaTeX
        monthly_recent = monthly.tail(12)
        
        latex_str = monthly_recent.to_latex(
            caption='Monthly Trends (Recent 12 Months)',
            label='tab:monthly_trends',
            index=False,
            column_format='l' + 'r' * (len(monthly_recent.columns) - 1),
            float_format='%.3f'
        )
        
        latex_file = output_dir / "monthly_trends.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        logger.info(f"Saved LaTeX table to {latex_file}")


def generate_top_bottom_companies_table(df, output_dir, format='csv', n=5):
    """
    Generate table of top and bottom performing companies.
    
    Args:
        df (pd.DataFrame): Results data
        output_dir (Path): Output directory
        format (str): Output format ('csv' or 'latex')
        n (int): Number of top/bottom companies to show
    """
    logger.info("Generating top/bottom companies table...")
    
    if 'company' not in df.columns or 'sentiment_score' not in df.columns:
        logger.warning("Missing required columns for top/bottom companies")
        return
    
    # Calculate company averages
    company_avg = df.groupby('company').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'rating': 'mean' if 'rating' in df.columns else lambda x: np.nan
    })
    
    # Flatten column names
    company_avg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in company_avg.columns.values]
    
    # Sort by sentiment
    company_avg = company_avg.sort_values('sentiment_score_mean', ascending=False)
    
    # Get top and bottom N
    top_companies = company_avg.head(n).copy()
    bottom_companies = company_avg.tail(n).copy()
    
    # Add rank column
    top_companies['Rank'] = range(1, len(top_companies) + 1)
    bottom_companies['Rank'] = range(len(company_avg) - len(bottom_companies) + 1, 
                                    len(company_avg) + 1)
    
    # Round values
    top_companies = top_companies.round(3)
    bottom_companies = bottom_companies.round(3)
    
    # Save top companies
    csv_file_top = output_dir / f"top_{n}_companies.csv"
    top_companies.to_csv(csv_file_top)
    logger.info(f"Saved top {n} companies to {csv_file_top}")
    
    # Save bottom companies
    csv_file_bottom = output_dir / f"bottom_{n}_companies.csv"
    bottom_companies.to_csv(csv_file_bottom)
    logger.info(f"Saved bottom {n} companies to {csv_file_bottom}")
    
    # Save to LaTeX if requested
    if format == 'latex':
        # Top companies
        latex_str_top = top_companies.to_latex(
            caption=f'Top {n} Companies by Average Sentiment Score',
            label='tab:top_companies',
            column_format='l' + 'r' * len(top_companies.columns),
            float_format='%.3f'
        )
        
        latex_file_top = output_dir / f"top_{n}_companies.tex"
        with open(latex_file_top, 'w') as f:
            f.write(latex_str_top)
        logger.info(f"Saved LaTeX table to {latex_file_top}")
        
        # Bottom companies
        latex_str_bottom = bottom_companies.to_latex(
            caption=f'Bottom {n} Companies by Average Sentiment Score',
            label='tab:bottom_companies',
            column_format='l' + 'r' * len(bottom_companies.columns),
            float_format='%.3f'
        )
        
        latex_file_bottom = output_dir / f"bottom_{n}_companies.tex"
        with open(latex_file_bottom, 'w') as f:
            f.write(latex_str_bottom)
        logger.info(f"Saved LaTeX table to {latex_file_bottom}")
    
    # Print to console
    print("\n" + "="*80)
    print(f"TOP {n} COMPANIES BY SENTIMENT")
    print("="*80)
    print(top_companies.to_string())
    print("="*80 + "\n")
    
    print("\n" + "="*80)
    print(f"BOTTOM {n} COMPANIES BY SENTIMENT")
    print("="*80)
    print(bottom_companies.to_string())
    print("="*80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate publication-quality tables from model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_tables.py --input data/results/model_results.csv --output-dir data/results/tables/
  python scripts/generate_tables.py --input data/results/model_results.csv --format latex
  python scripts/generate_tables.py --input data/results/model_results.csv --table-types summary,company,time
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input results CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save tables')
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'latex', 'both'],
                       help='Output format (default: csv)')
    parser.add_argument('--table-types', type=str, default='all',
                       help='Comma-separated list of table types (all, summary, company, '
                            'time, correlation, monthly, topbottom)')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of top/bottom companies to show (default: 5)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse table types
    if args.table_types == 'all':
        table_types = ['summary', 'company', 'time', 'correlation', 'monthly', 'topbottom']
    else:
        table_types = [tt.strip() for tt in args.table_types.split(',')]
    
    # Determine formats to generate
    if args.format == 'both':
        formats = ['csv', 'latex']
    else:
        formats = [args.format]
    
    logger.info("="*80)
    logger.info("TABLE GENERATION")
    logger.info("="*80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Table types: {', '.join(table_types)}")
    logger.info(f"Format: {args.format}")
    logger.info("="*80)
    
    try:
        # Load results
        df = load_results(args.input)
        
        # Generate requested tables
        for fmt in formats:
            if 'summary' in table_types:
                generate_summary_statistics_table(df, output_dir, format=fmt)
            
            if 'company' in table_types:
                generate_company_comparison_table(df, output_dir, format=fmt)
            
            if 'time' in table_types:
                generate_time_period_comparison_table(df, output_dir, format=fmt)
            
            if 'correlation' in table_types:
                generate_correlation_table(df, output_dir, format=fmt)
            
            if 'monthly' in table_types:
                generate_monthly_trends_table(df, output_dir, format=fmt)
            
            if 'topbottom' in table_types:
                generate_top_bottom_companies_table(df, output_dir, format=fmt, n=args.top_n)
        
        logger.info("="*80)
        logger.info("TABLE GENERATION COMPLETE!")
        logger.info(f"All tables saved to: {output_dir}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during table generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())