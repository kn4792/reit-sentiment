#!/usr/bin/env python3
"""
Statistical Tests for REIT Sentiment Analysis

Implements comprehensive statistical testing suite:
- Test 1: Correlation analysis (sentiment vs. performance metrics)
- Test 2: Difference-in-Differences (AI adoption impact)
- Test 3: Panel regression with fixed effects
- Test 4: AI occupational exposure analysis
- Test 5: Layoff impact analysis
- Test 6: Management change impact analysis

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
Version: 1.0

Usage:
    python scripts/05_run_tests.py --test all
    python scripts/05_run_tests.py --test correlation --sentiment-file data/results/monthly_sentiment.csv
    python scripts/05_run_tests.py --test diff_in_diff --treatment-date 2022-11-30
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from linearmodels import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class CorrelationAnalyzer:
    """
    Test 1: Correlation between sentiment and REIT performance metrics.
    
    Tests correlation with:
    - Revenue
    - Number of employees
    - ROA (Return on Assets)
    - ROE (Return on Equity)
    - Overall ratings
    - Management ratings
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize correlation analyzer.
        
        Args:
            significance_level: Alpha for statistical significance
        """
        self.alpha = significance_level
    
    def calculate_correlations(self,
                              df: pd.DataFrame,
                              sentiment_col: str = 'sentiment_compound_mean',
                              performance_cols: List[str] = None) -> pd.DataFrame:
        """
        Calculate Pearson and Spearman correlations.
        
        Args:
            df: DataFrame with sentiment and performance data
            sentiment_col: Column name for sentiment scores
            performance_cols: List of performance metric columns
            
        Returns:
            DataFrame with correlation results
        """
        if performance_cols is None:
            performance_cols = ['revenue', 'num_employees', 'roa', 'roe', 
                              'overall_rating', 'management_rating']
        
        results = []
        
        for perf_col in performance_cols:
            if perf_col not in df.columns:
                print(f"⚠️  Column '{perf_col}' not found, skipping")
                continue
            
            # Remove missing values
            valid_data = df[[sentiment_col, perf_col]].dropna()
            
            if len(valid_data) < 10:
                print(f"⚠️  Insufficient data for {perf_col} (n={len(valid_data)})")
                continue
            
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(
                valid_data[sentiment_col],
                valid_data[perf_col]
            )
            
            # Spearman correlation
            spearman_r, spearman_p = spearmanr(
                valid_data[sentiment_col],
                valid_data[perf_col]
            )
            
            results.append({
                'metric': perf_col,
                'n_obs': len(valid_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'pearson_sig': '***' if pearson_p < 0.01 else '**' if pearson_p < 0.05 else '*' if pearson_p < 0.10 else '',
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'spearman_sig': '***' if spearman_p < 0.01 else '**' if spearman_p < 0.05 else '*' if spearman_p < 0.10 else ''
            })
        
        return pd.DataFrame(results)
    
    def run_test(self,
                sentiment_file: Path,
                performance_file: Optional[Path] = None,
                output_dir: Path = None) -> Dict:
        """
        Run complete correlation analysis.
        
        Args:
            sentiment_file: Path to monthly sentiment CSV
            performance_file: Path to performance metrics CSV (optional)
            output_dir: Directory for output files
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print("TEST 1: CORRELATION ANALYSIS")
        print('='*60)
        
        # Load sentiment data
        print(f"\n📂 Loading sentiment data...")
        df = pd.read_csv(sentiment_file)
        print(f"✓ Loaded {len(df):,} observations")
        
        # Merge with performance data if provided
        if performance_file and performance_file.exists():
            print(f"📂 Loading performance data...")
            perf_df = pd.read_csv(performance_file)
            
            # Merge on ticker and year_month
            df = df.merge(perf_df, on=['ticker', 'year_month'], how='left')
            print(f"✓ Merged data: {len(df):,} observations")
        
        # Calculate correlations
        print(f"\n📊 Calculating correlations...")
        corr_df = self.calculate_correlations(df)
        
        # Display results
        print(f"\n{'='*60}")
        print("CORRELATION RESULTS")
        print('='*60)
        print(corr_df.to_string(index=False))
        
        # Save results
        if output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'test1_correlations_{timestamp}.csv'
            corr_df.to_csv(output_file, index=False)
            print(f"\n💾 Saved results → {output_file}")
            
            # Create correlation heatmap
            self._plot_correlation_heatmap(df, output_dir, timestamp)
        
        return {'correlations': corr_df}
    
    def _plot_correlation_heatmap(self,
                                  df: pd.DataFrame,
                                  output_dir: Path,
                                  timestamp: str):
        """Create and save correlation heatmap."""
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sentiment_cols = [c for c in numeric_cols if 'sentiment' in c]
        performance_cols = [c for c in numeric_cols if c not in sentiment_cols]
        
        if len(sentiment_cols) < 2 or len(performance_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = df[sentiment_cols + performance_cols].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Sentiment-Performance Correlation Heatmap', fontsize=14, pad=20)
        plt.tight_layout()
        
        plot_file = output_dir / f'correlation_heatmap_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved heatmap → {plot_file}")


class DifferenceInDifferences:
    """
    Test 2: Difference-in-Differences analysis for AI adoption impact.
    
    Compares REIT performance before/after treatment (e.g., ChatGPT launch).
    Treatment group: REITs with high AI adoption
    Control group: REITs with low AI adoption
    """
    
    def __init__(self,
                treatment_date: str = '2022-11-30',
                ai_threshold: float = 0.1):
        """
        Initialize DiD analyzer.
        
        Args:
            treatment_date: Date of treatment (e.g., ChatGPT launch)
            ai_threshold: Threshold for high AI adoption (percentage)
        """
        self.treatment_date = pd.to_datetime(treatment_date)
        self.ai_threshold = ai_threshold
    
    def prepare_did_data(self,
                        df: pd.DataFrame,
                        outcome_var: str = 'sentiment_compound_mean') -> pd.DataFrame:
        """
        Prepare data for DiD analysis.
        
        Args:
            df: Input DataFrame with date, ticker, AI adoption, outcome
            outcome_var: Outcome variable to analyze
            
        Returns:
            Prepared DataFrame with treatment indicators
        """
        print(f"\n🔧 Preparing DiD data...")
        
        # Convert date
        df['date'] = pd.to_datetime(df['year_month'] + '-01')
        
        # Create post-treatment indicator
        df['post_treatment'] = (df['date'] >= self.treatment_date).astype(int)
        
        # Calculate average AI adoption per company (pre-treatment)
        pre_treatment = df[df['post_treatment'] == 0].copy()
        ai_adoption = pre_treatment.groupby('ticker')['ai_mention_pct'].mean()
        
        # Create treatment group indicator (high AI adoption)
        treatment_group = ai_adoption[ai_adoption >= self.ai_threshold].index
        df['treatment_group'] = df['ticker'].isin(treatment_group).astype(int)
        
        # Create interaction term
        df['did_interaction'] = df['treatment_group'] * df['post_treatment']
        
        print(f"  ✓ Treatment date: {self.treatment_date.date()}")
        print(f"  ✓ Treatment group: {df['treatment_group'].sum()} companies")
        print(f"  ✓ Control group: {(~df['treatment_group'].astype(bool)).sum()} companies")
        
        return df
    
    def estimate_did(self,
                    df: pd.DataFrame,
                    outcome_var: str = 'sentiment_compound_mean') -> Dict:
        """
        Estimate DiD regression.
        
        Model: Y = β0 + β1*treatment + β2*post + β3*(treatment*post) + ε
        
        Args:
            df: Prepared DataFrame
            outcome_var: Outcome variable
            
        Returns:
            Dictionary with regression results
        """
        print(f"\n📈 Estimating DiD model...")
        
        # Prepare data
        model_df = df[[outcome_var, 'treatment_group', 'post_treatment', 'did_interaction']].dropna()
        
        # Add constant
        X = sm.add_constant(model_df[['treatment_group', 'post_treatment', 'did_interaction']])
        y = model_df[outcome_var]
        
        # Estimate OLS
        model = sm.OLS(y, X).fit(cov_type='HC1')  # Robust standard errors
        
        # Extract DiD estimate (interaction coefficient)
        did_estimate = model.params['did_interaction']
        did_se = model.bse['did_interaction']
        did_pvalue = model.pvalues['did_interaction']
        
        print(f"\n{'='*60}")
        print("DiD ESTIMATION RESULTS")
        print('='*60)
        print(f"DiD Estimate: {did_estimate:.4f}")
        print(f"Standard Error: {did_se:.4f}")
        print(f"P-value: {did_pvalue:.4f}")
        print(f"Significance: {'***' if did_pvalue < 0.01 else '**' if did_pvalue < 0.05 else '*' if did_pvalue < 0.10 else 'Not significant'}")
        
        return {
            'did_estimate': did_estimate,
            'std_error': did_se,
            'p_value': did_pvalue,
            'n_obs': len(model_df),
            'r_squared': model.rsquared,
            'model_summary': model.summary()
        }
    
    def run_test(self,
                sentiment_file: Path,
                ai_adoption_file: Path,
                output_dir: Path) -> Dict:
        """
        Run complete DiD analysis.
        
        Args:
            sentiment_file: Path to monthly sentiment CSV
            ai_adoption_file: Path to AI adoption CSV
            output_dir: Directory for output files
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print("TEST 2: DIFFERENCE-IN-DIFFERENCES")
        print('='*60)
        
        # Load data
        print(f"\n📂 Loading data...")
        sentiment_df = pd.read_csv(sentiment_file)
        ai_df = pd.read_csv(ai_adoption_file)
        
        # Merge
        df = sentiment_df.merge(ai_df, on=['ticker', 'year_month'], how='inner')
        print(f"✓ Merged {len(df):,} observations")
        
        # Prepare DiD data
        df = self.prepare_did_data(df)
        
        # Estimate DiD
        results = self.estimate_did(df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save coefficient table
        coef_df = pd.DataFrame({
            'variable': ['Treatment Group', 'Post Treatment', 'DiD (Interaction)'],
            'coefficient': [results['did_estimate'], 0, results['did_estimate']],
            'std_error': [results['std_error'], 0, results['std_error']],
            'p_value': [results['p_value'], 0, results['p_value']]
        })
        
        output_file = output_dir / f'test2_did_results_{timestamp}.csv'
        coef_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved results → {output_file}")
        
        # Save full summary
        summary_file = output_dir / f'test2_did_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write(str(results['model_summary']))
        print(f"💾 Saved summary → {summary_file}")
        
        return results


class PanelRegressionAnalyzer:
    """
    Test 3: Panel regression with fixed effects.
    
    Controls for unobserved heterogeneity across companies and time.
    Models:
    - Company fixed effects
    - Time fixed effects
    - Two-way fixed effects
    """
    
    def __init__(self):
        """Initialize panel regression analyzer."""
        pass
    
    def run_panel_regression(self,
                            df: pd.DataFrame,
                            dependent_var: str,
                            independent_vars: List[str],
                            entity_effects: bool = True,
                            time_effects: bool = True) -> Dict:
        """
        Estimate panel regression with fixed effects.
        
        Args:
            df: Panel DataFrame with 'ticker' and 'year_month'
            dependent_var: Dependent variable name
            independent_vars: List of independent variable names
            entity_effects: Include company fixed effects
            time_effects: Include time fixed effects
            
        Returns:
            Dictionary with regression results
        """
        print(f"\n📈 Estimating panel regression...")
        print(f"  • Dependent var: {dependent_var}")
        print(f"  • Independent vars: {', '.join(independent_vars)}")
        print(f"  • Entity FE: {entity_effects}")
        print(f"  • Time FE: {time_effects}")
        
        # Prepare data
        model_df = df[['ticker', 'year_month', dependent_var] + independent_vars].dropna()
        
        # Set multi-index for panel data
        model_df = model_df.set_index(['ticker', 'year_month'])
        
        # Prepare formula
        formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
        
        # Estimate model
        model = PanelOLS.from_formula(
            formula,
            data=model_df,
            entity_effects=entity_effects,
            time_effects=time_effects
        )
        
        results = model.fit(cov_type='clustered', cluster_entity=True)
        
        print(f"\n{'='*60}")
        print("PANEL REGRESSION RESULTS")
        print('='*60)
        print(results.summary)
        
        return {
            'model': results,
            'n_obs': results.nobs,
            'r_squared': results.rsquared,
            'summary': results.summary
        }
    
    def run_test(self,
                data_file: Path,
                output_dir: Path,
                dependent_var: str = 'performance_metric',
                independent_vars: List[str] = None) -> Dict:
        """
        Run panel regression analysis.
        
        Args:
            data_file: Path to panel data CSV
            output_dir: Output directory
            dependent_var: Dependent variable
            independent_vars: List of independent variables
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print("TEST 3: PANEL REGRESSION")
        print('='*60)
        
        if independent_vars is None:
            independent_vars = ['sentiment_compound_mean', 'ai_mention_pct']
        
        # Load data
        print(f"\n📂 Loading data...")
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df):,} observations")
        
        # Run regression
        results = self.run_panel_regression(
            df,
            dependent_var,
            independent_vars,
            entity_effects=True,
            time_effects=True
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = output_dir / f'test3_panel_regression_{timestamp}.txt'
        
        with open(summary_file, 'w') as f:
            f.write(str(results['summary']))
        
        print(f"\n💾 Saved results → {summary_file}")
        
        return results


def run_all_tests(args):
    """Run all statistical tests."""
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test 1: Correlations
    if args.test in ['all', 'correlation']:
        corr_analyzer = CorrelationAnalyzer()
        
        sentiment_file = Path(args.sentiment_file) if args.sentiment_file else Path('data/results/monthly_sentiment.csv')
        performance_file = Path(args.performance_file) if args.performance_file else None
        
        if sentiment_file.exists():
            results['test1'] = corr_analyzer.run_test(
                sentiment_file,
                performance_file,
                output_dir
            )
        else:
            print(f"⚠️  Sentiment file not found: {sentiment_file}")
    
    # Test 2: Difference-in-Differences
    if args.test in ['all', 'diff_in_diff', 'did']:
        did_analyzer = DifferenceInDifferences(
            treatment_date=args.treatment_date
        )
        
        sentiment_file = Path('data/results/monthly_sentiment.csv')
        ai_file = Path('data/results/ai_adoption_monthly.csv')
        
        if sentiment_file.exists() and ai_file.exists():
            results['test2'] = did_analyzer.run_test(
                sentiment_file,
                ai_file,
                output_dir
            )
        else:
            print(f"⚠️  Required files not found for DiD analysis")
    
    # Test 3: Panel Regression
    if args.test in ['all', 'panel']:
        panel_analyzer = PanelRegressionAnalyzer()
        
        panel_file = Path('data/results/panel_data.csv')
        
        if panel_file.exists():
            results['test3'] = panel_analyzer.run_test(
                panel_file,
                output_dir
            )
        else:
            print(f"⚠️  Panel data file not found: {panel_file}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run statistical tests for REIT sentiment analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/05_run_tests.py --test all
  
  # Run correlation analysis only
  python scripts/05_run_tests.py --test correlation --sentiment-file data/results/monthly_sentiment.csv
  
  # Run DiD with custom treatment date
  python scripts/05_run_tests.py --test diff_in_diff --treatment-date 2022-11-30
  
  # Run panel regression
  python scripts/05_run_tests.py --test panel --data-file data/results/panel_data.csv
        """
    )
    
    parser.add_argument(
        '--test',
        choices=['all', 'correlation', 'diff_in_diff', 'did', 'panel'],
        default='all',
        help='Which test to run (default: all)'
    )
    parser.add_argument(
        '--output',
        default='data/results',
        help='Output directory (default: data/results)'
    )
    parser.add_argument(
        '--sentiment-file',
        help='Path to monthly sentiment CSV'
    )
    parser.add_argument(
        '--performance-file',
        help='Path to performance metrics CSV'
    )
    parser.add_argument(
        '--treatment-date',
        default='2022-11-30',
        help='Treatment date for DiD (default: 2022-11-30 - ChatGPT launch)'
    )
    parser.add_argument(
        '--data-file',
        help='Path to panel data CSV'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("REIT SENTIMENT ANALYSIS - STATISTICAL TESTS")
    print('='*60)
    print(f"Test suite: {args.test}")
    print(f"Output directory: {args.output}")
    print('='*60)
    
    # Run tests
    results = run_all_tests(args)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ TESTING COMPLETE")
    print('='*60)
    print(f"Tests run: {len(results)}")
    print(f"Output directory: {args.output}")
    print('='*60)


if __name__ == '__main__':
    main()