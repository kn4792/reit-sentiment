"""
Panel Regression Module

Panel data analysis with fixed effects for REIT sentiment analysis.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from typing import List, Dict


class PanelRegressionAnalyzer:
    """
    Panel regression with fixed effects.
    
    Controls for unobserved heterogeneity across companies and time.
    Models:
    - Company fixed effects
    - Time fixed effects
    - Two-way fixed effects
    
    Example:
        >>> analyzer = PanelRegressionAnalyzer()
        >>> results = analyzer.run_regression(df, dependent_var='performance',
        ...                                   independent_vars=['sentiment'])
    """
    
    def __init__(self):
        """Initialize panel regression analyzer."""
        pass
    
    def run_regression(self,
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
        
        return {
            'model': results,
            'n_obs': results.nobs,
            'r_squared': results.rsquared,
            'coefficients': results.params.to_dict(),
            'std_errors': results.std_errors.to_dict(),
            'p_values': results.pvalues.to_dict()
        }


if __name__ == '__main__':
    print("Panel Regression Example")
    print("=" * 60)
    
    # Sample panel data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-01', freq='M')
    tickers = ['PLD', 'AMT', 'EQIX']
    
    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'ticker': ticker,
                'year_month': date.strftime('%Y-%m'),
                'performance': np.random.normal(100, 20),
                'sentiment_compound_mean': np.random.normal(0.2, 0.3),
                'ai_mention_pct': np.random.uniform(0, 20)
            })
    
    df = pd.DataFrame(data)
    
    # Run regression
    analyzer = PanelRegressionAnalyzer()
    results = analyzer.run_regression(
        df,
        dependent_var='performance',
        independent_vars=['sentiment_compound_mean', 'ai_mention_pct']
    )
    
    print("\nPanel Regression Results:")
    print(f"  N observations: {results['n_obs']}")
    print(f"  R-squared: {results['r_squared']:.4f}")
    print("\n  Coefficients:")
    for var, coef in results['coefficients'].items():
        print(f"    {var}: {coef:.4f}")