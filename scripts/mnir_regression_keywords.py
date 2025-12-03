#!/usr/bin/env python3
"""
MNIR Stage 2: Regression and Index Construction

Performs the core Multinomial Inverse Regression (MNIR) analysis:
1. Runs Poisson regression for EACH word in the vocabulary:
   Word_Count_j ~ GenAI_Intensity + Year_FE + Controls
2. Extracts the coefficient (φ_j) for GenAI_Intensity for each word.
3. Filters coefficients by statistical significance (|t| > 1.96)
4. Constructs the "GenAI Productivity Index" for each firm-year by weighting
   word frequencies by their learned coefficients.

SIMPLIFIED VERSION (v3.0):
- Uses continuous GenAI_Intensity instead of binary treatment
- Measures GenAI-related language productivity
- Includes year fixed effects for time trends
- Filters by statistical significance
- Fixed index construction (normalize before dot product)

Methodology based on Campbell, Shang (2021) "Tone at the Bottom".

Output Files:
- word_weights.csv: Learned coefficients (φ) and t-statistics for each word
- genai_productivity_index.csv: Final firm-year measure of GenAI productivity
- regression_stats.json: Summary of model performance
- validation_report.txt: Data alignment and quality checks

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-11-28

Usage:
    python scripts/mnir_regression_keywords.py
    python scripts/mnir_regression_keywords.py --min-t-stat 1.96
    python scripts/mnir_regression_keywords.py --no-fixed-effects  # for debugging
"""

import sys
import argparse
import warnings
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


def run_single_regression(args: Tuple) -> Dict:
    """
    Run Poisson regression for a single word.
    
    Model: E[Word_Count] = exp(α + β*Controls + δ*Year_FE + φ*GenAI_Intensity + log(Total_Words))
    
    Args:
        args: Tuple containing (word, counts, covariates, offset)
        
    Returns:
        Dictionary with regression results
    """
    word, y, X, offset = args
    
    try:
        # Check for zero variance
        if y.std() == 0:
            return {
                'word': word,
                'coef': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'converged': False,
                'mean_count': float(np.mean(y)),
                'std_count': 0.0,
                'status': 'zero_variance'
            }
        
        # Fit Poisson GLM
        model = sm.GLM(y, X, family=Poisson(), offset=offset)
        result = model.fit(maxiter=100, disp=False)
        
        # Extract results for genai_intensity (our variable of interest)
        if 'genai_intensity' not in result.params.index:
            return {
                'word': word,
                'coef': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'converged': False,
                'mean_count': float(np.mean(y)),
                'std_count': float(np.std(y)),
                'status': 'missing_genai_intensity'
            }
        
        coef = result.params['genai_intensity']
        t_stat = result.tvalues['genai_intensity']
        p_val = result.pvalues['genai_intensity']
        converged = result.converged
        
        return {
            'word': word,
            'coef': float(coef),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'converged': bool(converged),
            'mean_count': float(np.mean(y)),
            'std_count': float(np.std(y)),
            'status': 'success' if converged else 'not_converged'
        }
        
    except Exception as e:
        return {
            'word': word,
            'coef': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'converged': False,
            'mean_count': float(np.mean(y)),
            'std_count': float(np.std(y)) if len(y) > 1 else 0.0,
            'status': f'error: {str(e)[:100]}'
        }


class MNIRRegression:
    """
    Manages MNIR regression and index construction.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, use_fixed_effects: bool = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_fixed_effects = use_fixed_effects
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.firm_year_df = None
        self.word_counts = {}
        self.vocabulary = []
        self.results = []
        self.validation_log = []
    
    def log_validation(self, message: str):
        """Log validation messages."""
        print(f"  {message}")
        self.validation_log.append(message)
    
    def load_data(self):
        """Load preprocessed data with validation."""
        print(f"\n{'='*70}")
        print("LOADING DATA")
        print('='*70)
        
        # Load firm-year data (controls + outcome)
        firm_year_file = self.input_dir / 'firm_year_data.csv'
        if not firm_year_file.exists():
            raise FileNotFoundError(f"Missing: {firm_year_file}")
        
        self.firm_year_df = pd.read_csv(firm_year_file)
        self.log_validation(f"Loaded {len(self.firm_year_df):,} firm-year observations")
        
        # Validate required columns
        required_cols = ['ticker', 'year', 'genai_intensity', 'rating', 'review_count',
                        'word_count_pros', 'word_count_cons']
        missing = [c for c in required_cols if c not in self.firm_year_df.columns]
        if missing:
            raise ValueError(f"Missing columns in firm_year_data.csv: {missing}")
        
        # Load vocabulary
        vocab_file = self.input_dir / 'vocabulary.json'
        if not vocab_file.exists():
            raise FileNotFoundError(f"Missing: {vocab_file}")
        
        with open(vocab_file, 'r') as f:
            self.vocabulary = json.load(f)
        self.log_validation(f"Loaded {len(self.vocabulary):,} vocabulary words")
        
        # Load word counts
        pros_file = self.input_dir / 'word_counts_pros.csv'
        cons_file = self.input_dir / 'word_counts_cons.csv'
        
        if not pros_file.exists():
            raise FileNotFoundError(f"Missing: {pros_file}")
        if not cons_file.exists():
            raise FileNotFoundError(f"Missing: {cons_file}")
        
        self.word_counts['pros'] = pd.read_csv(pros_file)
        self.word_counts['cons'] = pd.read_csv(cons_file)
        self.log_validation(f"Loaded word count matrices")
        
        # Validate data alignment
        self.validate_data_alignment()
    
    def validate_data_alignment(self):
        """Ensure firm_year_data aligns with word count matrices."""
        print(f"\n{'='*70}")
        print("VALIDATING DATA ALIGNMENT")
        print('='*70)
        
        for section in ['pros', 'cons']:
            counts = self.word_counts[section]
            
            # Check dimensions
            self.log_validation(f"{section.upper()}: {counts.shape[0]} rows × {counts.shape[1]} columns")
            
            # Check for matching rows
            merged = self.firm_year_df.merge(
                counts[['ticker', 'year']],
                on=['ticker', 'year'],
                how='inner',
                indicator=True
            )
            
            n_matched = (merged['_merge'] == 'both').sum()
            n_left_only = (merged['_merge'] == 'left_only').sum()
            
            if n_matched != len(self.firm_year_df):
                self.log_validation(
                    f"WARNING: {section} has {n_left_only} unmatched firm-years in firm_year_data"
                )
            else:
                self.log_validation(f"{section}: All {n_matched:,} firm-years matched")
            
            # Check for word columns with 'wc_' prefix
            word_cols = [c for c in counts.columns if c.startswith('wc_')]
            self.log_validation(f"{section}: Found {len(word_cols):,} word count columns")
            
            # Verify no missing values in critical columns
            if counts[['ticker', 'year']].isna().any().any():
                self.log_validation(f"WARNING: {section} has missing ticker/year values")
        
        # Validate GenAI intensity distribution
        print(f"\nGenAI Intensity Distribution:")
        print(f"    Mean: {self.firm_year_df['genai_intensity'].mean():.4f}")
        print(f"    Std:  {self.firm_year_df['genai_intensity'].std():.4f}")
        print(f"    Min:  {self.firm_year_df['genai_intensity'].min():.4f}")
        print(f"    Max:  {self.firm_year_df['genai_intensity'].max():.4f}")
    
    def prepare_regression_data(self, section: str = 'pros') -> Tuple:
        """
        Prepare matrices for regression with year fixed effects.
        
        Args:
            section: 'pros' or 'cons'
            
        Returns:
            (df, X, offset, word_columns)
        """
        # Merge counts with firm-year data to ensure alignment
        df = self.firm_year_df.merge(
            self.word_counts[section],
            on=['ticker', 'year'],
            how='inner'
        )
        
        if len(df) == 0:
            raise ValueError(f"No matching rows after merge for {section}")
        
        # Define base covariates
        df['intercept'] = 1.0
        df['log_reviews'] = np.log1p(df['review_count'])
        
        # CRITICAL FIX: Scale genai_intensity to percentage (0-100) for numerical stability
        # Original values are very small (mean ~0.006), causing convergence issues
        df['genai_intensity_pct'] = df['genai_intensity'] * 100
        
        print(f"    GenAI Intensity (scaled to %):")
        print(f"      Mean: {df['genai_intensity_pct'].mean():.4f}%")
        print(f"      Std:  {df['genai_intensity_pct'].std():.4f}%")
        print(f"      Range: [{df['genai_intensity_pct'].min():.4f}%, {df['genai_intensity_pct'].max():.4f}%]")
        
        if self.use_fixed_effects:
            # Use YEAR fixed effects to control for time trends
            # CRITICAL: Convert year to integer first to avoid dtype issues
            df['year'] = df['year'].astype(int)
            year_dummies = pd.get_dummies(df['year'], prefix='year', drop_first=True, dtype=float)
            
            # Combine: base covariates + year dummies
            # Use SCALED genai_intensity for numerical stability
            X = pd.concat([
                df[['intercept', 'rating', 'log_reviews', 'genai_intensity_pct']],
                year_dummies
            ], axis=1)
            
            # Rename column for clarity in results
            X.rename(columns={'genai_intensity_pct': 'genai_intensity'}, inplace=True)
            
            # Ensure all columns are numeric (float64)
            X = X.astype(float)
            
            print(f"    Added {year_dummies.shape[1]} year fixed effects")
        else:
            X = df[['intercept', 'rating', 'log_reviews', 'genai_intensity_pct']].copy()
            X.rename(columns={'genai_intensity_pct': 'genai_intensity'}, inplace=True)
            X = X.astype(float)
        
        # Offset: log(Total_Word_Count)
        total_words_col = f'word_count_{section}'
        offset = np.log(df[total_words_col] + 1).values
        
        # Get word columns (those with 'wc_' prefix)
        word_columns = [c for c in df.columns if c.startswith('wc_')]
        
        return df, X, offset, word_columns
    
    def run_regressions(self, section: str = 'pros', n_jobs: int = 1):
        """
        Run regressions for all words.
        
        Args:
            section: 'pros' or 'cons'
            n_jobs: Number of CPU cores (default: 1 for stability)
        """
        print(f"\n{'='*70}")
        print(f"RUNNING REGRESSIONS: {section.upper()}")
        print('='*70)
        
        # Prepare data
        df, X, offset, word_columns = self.prepare_regression_data(section)
        
        print(f"  Data shape: {df.shape}")
        print(f"  Covariates: {X.shape[1]} variables")
        print(f"  Words to process: {len(word_columns):,}")
        if self.use_fixed_effects:
            print(f"Using year fixed effects")
        else:
            print(f"No fixed effects")
        
        # Prepare arguments for processing
        tasks = []
        for word_col in word_columns:
            # Extract original word (remove 'wc_' prefix)
            word = word_col[3:]  # Remove 'wc_'
            y = df[word_col].values
            tasks.append((word, y, X, offset))
        
        print(f"\n  Processing {len(tasks):,} words...")
        
        # Run regressions (sequential for stability)
        results = []
        for task in tqdm(tasks, desc=f"  Regressing"):
            res = run_single_regression(task)
            results.append(res)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df['section'] = section
        
        # Print detailed diagnostics
        print(f"\nDetailed Status Breakdown:")
        status_counts = results_df['status'].value_counts()
        for status, count in status_counts.items():
            print(f"- {status}: {count:,} ({count/len(results_df)*100:.1f}%)")
        
        # Print summary
        n_converged = (results_df['converged'] == True).sum()
        n_significant = ((results_df['t_stat'].abs() > 1.96) & 
                        (results_df['converged'] == True)).sum()
        
        print(f"\nRegressions complete:")
        print(f"- Converged: {n_converged:,} / {len(results_df):,} ({n_converged/len(results_df)*100:.1f}%)")
        print(f"- Significant (|t| > 1.96): {n_significant:,} ({n_significant/len(results_df)*100:.1f}%)")
        
        # If convergence is very low, print first few errors as examples
        if n_converged < len(results_df) * 0.1:  # Less than 10% convergence
            print(f"\nLow convergence rate! Sample errors:")
            error_samples = results_df[results_df['converged'] == False].head(3)
            for idx, row in error_samples.iterrows():
                print(f"Word: {row['word']}, Status: {row['status']}")
        
        return results_df
    
    def construct_index(self, weights_df: pd.DataFrame, section: str = 'pros',
                       min_t_stat: float = 1.96) -> np.ndarray:
        """
        Construct GenAI Productivity Index using vectorized operations.
        
        Index = (Normalized_Counts) @ Weights
        where Normalized_Counts = Counts / Total_Words
        
        Args:
            weights_df: DataFrame with regression coefficients
            section: 'pros' or 'cons'
            min_t_stat: Minimum |t-statistic| for inclusion (default: 1.96)
            
        Returns:
            Array of index scores (aligned with firm_year_df)
        """
        print(f"\n{'='*70}")
        print(f"CONSTRUCTING INDEX: {section.upper()}")
        print('='*70)
        
        # Filter weights by convergence and significance
        valid_weights = weights_df[
            (weights_df['converged'] == True) & 
            (weights_df['t_stat'].abs() >= min_t_stat)
        ].copy()
        
        if len(valid_weights) == 0:
            print(f"WARNING: No significant weights found for {section}")
            return np.zeros(len(self.firm_year_df))
        
        print(f"Using {len(valid_weights):,} significant weights (|t| ≥ {min_t_stat})")
        print(f"Mean |coefficient|: {valid_weights['coef'].abs().mean():.4f}")
        
        # Create word -> coefficient mapping
        word_to_coef = valid_weights.set_index('word')['coef'].to_dict()
        
        # Get count matrix and align
        counts_df = self.word_counts[section].copy()
        counts_df = counts_df.set_index(['ticker', 'year'])
        
        # Build aligned matrices
        words_to_use = []
        for word in valid_weights['word']:
            word_col = f'wc_{word}'
            if word_col in counts_df.columns:
                words_to_use.append(word)
        
        if len(words_to_use) == 0:
            print(f"WARNING: No word columns match between counts and weights")
            return np.zeros(len(self.firm_year_df))
        
        print(f"Matched {len(words_to_use):,} words between counts and weights")
        
        # Extract count matrix for matched words
        word_cols = [f'wc_{w}' for w in words_to_use]
        X = counts_df[word_cols].copy()
        
        # Get weights in same order
        w = np.array([word_to_coef[word] for word in words_to_use])
        
        # Get total words for normalization
        total_words = self.firm_year_df.set_index(['ticker', 'year'])[f'word_count_{section}']
        total_words = total_words.reindex(X.index).fillna(1)
        total_words = total_words.replace(0, 1)
        
        # CRITICAL: Normalize BEFORE dot product
        X_normalized = X.div(total_words, axis=0)
        
        # Calculate index: weighted sum of normalized frequencies
        scores = X_normalized.dot(w)
        scores = scores.fillna(0.0)
        
        print(f"Index constructed")
        print(f"Mean: {scores.mean():.6f}")
        print(f"Std:  {scores.std():.6f}")
        print(f"Min:  {scores.min():.6f}")
        print(f"Max:  {scores.max():.6f}")
        
        return scores.values
    
    def save_validation_report(self):
        """Save validation log to file."""
        report_file = self.output_dir / 'validation_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MNIR REGRESSION VALIDATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("\n")
            for msg in self.validation_log:
                f.write(msg + "\n")
        
        print(f"Saved validation report → {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='MNIR Regression Stage (v3.0 - continuous GenAI measure)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: with year fixed effects and significance filtering
  python scripts/mnir_regression_v3.py
  
  # Custom significance threshold
  python scripts/mnir_regression_v3.py --min-t-stat 2.58  # 99% confidence
  
  # Disable fixed effects (for debugging)
  python scripts/mnir_regression_v3.py --no-fixed-effects
  
  # Custom input/output directories
  python scripts/mnir_regression_v3.py --input data/processed/mnir --output data/results/mnir_v3
        """
    )
    parser.add_argument(
        '--input', 
        default='data/processed/mnir', 
        help='Input directory with preprocessed data'
    )
    parser.add_argument(
        '--output', 
        default='data/results/mnir', 
        help='Output directory'
    )
    parser.add_argument(
        '--min-t-stat', 
        type=float, 
        default=1.96, 
        help='Minimum |t-statistic| for index construction (default: 1.96 = 95%% confidence)'
    )
    parser.add_argument(
        '--no-fixed-effects',
        action='store_true',
        help='Disable year fixed effects (for debugging)'
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print("="*70)
    print("MNIR STAGE 2: REGRESSION ANALYSIS (v3.0)")
    print("="*70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Min |t-stat|: {args.min_t_stat}")
    print(f"Year fixed effects: {'Disabled' if args.no_fixed_effects else 'Enabled'}")
    
    # Initialize
    mnir = MNIRRegression(
        input_dir, 
        output_dir, 
        use_fixed_effects=not args.no_fixed_effects
    )
    
    try:
        mnir.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Run Regressions
    try:
        pros_results = mnir.run_regressions('pros')
        cons_results = mnir.run_regressions('cons')
    except Exception as e:
        print(f"Error in regressions: {e}")
        return 1
    
    # Combine and save weights
    all_weights = pd.concat([pros_results, cons_results], ignore_index=True)
    weights_file = output_dir / 'word_weights.csv'
    all_weights.to_csv(weights_file, index=False)
    print(f"Saved word weights → {weights_file}")
    
    # Construct Indices
    firm_year_scores = mnir.firm_year_df[['ticker', 'year']].copy()
    
    try:
        firm_year_scores['mnir_score_pros'] = mnir.construct_index(
            pros_results, 'pros', min_t_stat=args.min_t_stat
        )
        firm_year_scores['mnir_score_cons'] = mnir.construct_index(
            cons_results, 'cons', min_t_stat=args.min_t_stat
        )
    except Exception as e:
        print(f"Error constructing indices: {e}")
        return 1
    
    # Combined Score (Average of pros and cons)
    firm_year_scores['genai_productivity_index'] = (
        firm_year_scores['mnir_score_pros'] + firm_year_scores['mnir_score_cons']
    ) / 2
    
    # Save Index
    index_file = output_dir / 'genai_productivity_index.csv'
    firm_year_scores.to_csv(index_file, index=False)
    print(f"Saved GenAI Productivity Index → {index_file}")
    
    # Save validation report
    mnir.save_validation_report()
    
    # Detailed Analysis
    print("="*70)
    print("INDEX STATISTICS")
    print("="*70)
    
    for col in ['mnir_score_pros', 'mnir_score_cons', 'genai_productivity_index']:
        values = firm_year_scores[col]
        print(f"\n{col}:")
        print(f"  Mean:   {values.mean():10.6f}")
        print(f"  Median: {values.median():10.6f}")
        print(f"  Std:    {values.std():10.6f}")
        print(f"  Min:    {values.min():10.6f}")
        print(f"  Max:    {values.max():10.6f}")
        print(f"  Non-zero: {(values != 0).sum():,} / {len(values):,} ({(values != 0).mean()*100:.1f}%)")
    
    # Top Words Analysis
    print("="*70)
    print("TOP PREDICTIVE WORDS")
    print("="*70)
    
    for section in ['pros', 'cons']:
        print(f"\n{section.upper()} Section (Positive Association with GenAI):")
        top_words = all_weights[
            (all_weights['section'] == section) & 
            (all_weights['converged'] == True) &
            (all_weights['t_stat'].abs() >= args.min_t_stat)
        ].sort_values('coef', ascending=False).head(15)
        
        if len(top_words) == 0:
            print("  (No significant words)")
        else:
            for i, row in enumerate(top_words.itertuples(), 1):
                print(f"  {i:2d}. {row.word:<20} coef: {row.coef:8.4f}  t: {row.t_stat:7.2f}  p: {row.p_value:.4f}")
    
    print("="*70)
    print("BOTTOM WORDS")
    print("="*70)
    
    for section in ['pros', 'cons']:
        print(f"\n{section.upper()} Section (Negative Association with GenAI):")
        bot_words = all_weights[
            (all_weights['section'] == section) & 
            (all_weights['converged'] == True) &
            (all_weights['t_stat'].abs() >= args.min_t_stat)
        ].sort_values('coef', ascending=True).head(15)
        
        if len(bot_words) == 0:
            print("  (No significant words)")
        else:
            for i, row in enumerate(bot_words.itertuples(), 1):
                print(f"  {i:2d}. {row.word:<20} coef: {row.coef:8.4f}  t: {row.t_stat:7.2f}  p: {row.p_value:.4f}")
    
    # Summary statistics
    print("="*70)
    print("REGRESSION SUMMARY")
    print("="*70)
    
    total_words = len(all_weights)
    converged = (all_weights['converged'] == True).sum()
    significant = ((all_weights['t_stat'].abs() >= args.min_t_stat) & 
                  (all_weights['converged'] == True)).sum()
    
    print(f"\nTotal words processed: {total_words:,}")
    print(f"Converged: {converged:,} ({converged/total_words*100:.1f}%)")
    print(f"Significant (|t| ≥ {args.min_t_stat}): {significant:,} ({significant/total_words*100:.1f}%)")
    
    # Distribution of coefficients
    sig_weights = all_weights[
        (all_weights['converged'] == True) & 
        (all_weights['t_stat'].abs() >= args.min_t_stat)
    ]
    
    if len(sig_weights) > 0:
        pos_coefs = (sig_weights['coef'] > 0).sum()
        neg_coefs = (sig_weights['coef'] < 0).sum()
        
        print(f"\nSignificant coefficients:")
        print(f"  Positive: {pos_coefs:,} ({pos_coefs/len(sig_weights)*100:.1f}%)")
        print(f"  Negative: {neg_coefs:,} ({neg_coefs/len(sig_weights)*100:.1f}%)")
        print(f"  Mean |coef|: {sig_weights['coef'].abs().mean():.4f}")
    
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output files:")
    print(f"  {weights_file.name}")
    print(f"  {index_file.name}")
    print(f"  validation_report.txt")
    print(f"\nNEXT STEPS:")
 print(f"  1. Review validation_report.txt for data quality")
    print(f"  2. Run: python scripts/map_stems_to_words.py")
    print(f"  3. Examine word_weights_readable.csv for interpretable words")
    print(f"  4. Run: python scripts/analysis.py")
    print(f"  5. Merge chatgpt_language_index.csv with REIT performance data")
    print(f"  6. Test hypothesis: High ChatGPT Language Index → Better performance?")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    # Windows support for multiprocessing
    mp.freeze_support()
    sys.exit(main())