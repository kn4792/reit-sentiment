"""
Correlation Tests Module

Test correlation between sentiment and REIT performance metrics.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelationAnalyzer:
    """
    Analyze correlations between sentiment and performance metrics.
    
    Tests correlation with:
    - Revenue
    - Number of employees
    - ROA/ROE
    - Overall ratings
    
    Example:
        >>> analyzer = CorrelationAnalyzer()
        >>> results = analyzer.run_analysis(df)
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
                continue
            
            # Remove missing values
            valid_data = df[[sentiment_col, perf_col]].dropna()
            
            if len(valid_data) < 10:
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
    
    def plot_correlation_heatmap(self,
                                df: pd.DataFrame,
                                output_path: str = None):
        """
        Create correlation heatmap.
        
        Args:
            df: DataFrame with sentiment and performance columns
            output_path: Path to save plot (optional)
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Sentiment-Performance Correlation Heatmap', fontsize=14, pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        plt.close()


if __name__ == '__main__':
    print("Correlation Analysis Example")
    print("=" * 60)
    
    # Sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'sentiment_compound_mean': np.random.normal(0.2, 0.3, 100),
        'revenue': np.random.normal(1000, 200, 100),
        'roa': np.random.normal(0.05, 0.02, 100),
        'overall_rating': np.random.uniform(3, 5, 100)
    })
    
    analyzer = CorrelationAnalyzer()
    results = analyzer.calculate_correlations(df)
    print("\nCorrelation Results:")
    print(results)