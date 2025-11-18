"""
Difference-in-Differences Analysis Module

Analyze AI adoption impact using DiD methodology.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict


class DifferenceInDifferences:
    """
    Difference-in-Differences analysis for AI adoption impact.
    
    Compares REIT performance before/after treatment (e.g., ChatGPT launch).
    Treatment group: REITs with high AI adoption
    Control group: REITs with low AI adoption
    
    Example:
        >>> did = DifferenceInDifferences(treatment_date='2022-11-30')
        >>> results = did.estimate(df)
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
    
    def prepare_data(self,
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
        
        return df
    
    def estimate(self,
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
        # Prepare data
        model_df = df[[outcome_var, 'treatment_group', 'post_treatment', 'did_interaction']].dropna()
        
        # Add constant
        X = sm.add_constant(model_df[['treatment_group', 'post_treatment', 'did_interaction']])
        y = model_df[outcome_var]
        
        # Estimate OLS with robust standard errors
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        # Extract DiD estimate (interaction coefficient)
        did_estimate = model.params['did_interaction']
        did_se = model.bse['did_interaction']
        did_pvalue = model.pvalues['did_interaction']
        
        return {
            'did_estimate': did_estimate,
            'std_error': did_se,
            'p_value': did_pvalue,
            'n_obs': len(model_df),
            'r_squared': model.rsquared,
            'model': model
        }


if __name__ == '__main__':
    print("Difference-in-Differences Example")
    print("=" * 60)
    
    # Sample data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-01', freq='M')
    tickers = ['PLD', 'AMT', 'EQIX']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'ticker': ticker,
                'year_month': date.strftime('%Y-%m'),
                'ai_mention_pct': np.random.uniform(0, 20),
                'sentiment_compound_mean': np.random.normal(0.2, 0.3)
            })
    
    df = pd.DataFrame(data)
    
    # Run DiD
    did = DifferenceInDifferences(treatment_date='2022-11-30')
    df = did.prepare_data(df)
    results = did.estimate(df)
    
    print("\nDiD Results:")
    print(f"  DiD Estimate: {results['did_estimate']:.4f}")
    print(f"  Std Error: {results['std_error']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  R-squared: {results['r_squared']:.4f}")