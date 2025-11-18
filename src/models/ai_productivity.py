"""
AI Productivity Measurement Module

Measures AI adoption impact on REIT productivity and performance.

Author: Konain Niaz (kn4792@rit.edu)

"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy import stats


class AIProductivityAnalyzer:
    """
    Analyze AI adoption impact on REIT productivity.
    
    Measures:
    - AI adoption rate over time
    - Productivity changes pre/post AI adoption
    - AI occupational exposure effects
    
    Example:
        >>> analyzer = AIProductivityAnalyzer()
        >>> metrics = analyzer.calculate_productivity_metrics(df)
    """
    
    def __init__(self, ai_threshold: float = 0.1):
        """
        Initialize AI productivity analyzer.
        
        Args:
            ai_threshold: Threshold for high AI adoption (%)
        """
        self.ai_threshold = ai_threshold
    
    def calculate_adoption_rate(self,
                                df: pd.DataFrame,
                                time_column: str = 'year_month') -> pd.DataFrame:
        """
        Calculate AI adoption rate over time.
        
        Args:
            df: DataFrame with ai_mentioned column
            time_column: Time period column
            
        Returns:
            DataFrame with adoption rates
        """
        adoption = df.groupby(time_column).agg({
            'ai_mentioned': 'sum',
            'ticker': 'count'
        }).reset_index()
        
        adoption.rename(columns={'ticker': 'total_reviews'}, inplace=True)
        adoption['adoption_rate'] = (adoption['ai_mentioned'] / adoption['total_reviews']) * 100
        
        return adoption
    
    def classify_ai_adopters(self,
                            df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify REITs as high/low AI adopters.
        
        Args:
            df: DataFrame with AI mentions
            
        Returns:
            DataFrame with adopter classification
        """
        # Calculate average AI adoption per company
        company_ai = df.groupby('ticker').agg({
            'ai_mentioned': 'sum',
            'ticker': 'count'
        }).reset_index()
        
        company_ai.rename(columns={'ticker': 'total_reviews'}, inplace=True)
        company_ai['ai_adoption_pct'] = (company_ai['ai_mentioned'] / company_ai['total_reviews']) * 100
        
        # Classify
        company_ai['ai_adopter'] = company_ai['ai_adoption_pct'] >= self.ai_threshold
        company_ai['adopter_type'] = company_ai['ai_adopter'].map({
            True: 'High AI Adopter',
            False: 'Low AI Adopter'
        })
        
        return company_ai
    
    def measure_productivity_change(self,
                                   df: pd.DataFrame,
                                   treatment_date: str) -> Dict:
        """
        Measure productivity change before/after treatment.
        
        Args:
            df: DataFrame with sentiment and performance metrics
            treatment_date: Date of treatment (e.g., ChatGPT launch)
            
        Returns:
            Dictionary with productivity metrics
        """
        df['date'] = pd.to_datetime(df['date'])
        treatment_dt = pd.to_datetime(treatment_date)
        
        # Split data
        pre = df[df['date'] < treatment_dt]
        post = df[df['date'] >= treatment_dt]
        
        # Calculate metrics
        metrics = {
            'pre_mean_sentiment': pre['sentiment_compound'].mean(),
            'post_mean_sentiment': post['sentiment_compound'].mean(),
            'sentiment_change': post['sentiment_compound'].mean() - pre['sentiment_compound'].mean(),
            'pre_reviews': len(pre),
            'post_reviews': len(post)
        }
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(post['sentiment_compound'], pre['sentiment_compound'])
        metrics['t_statistic'] = t_stat
        metrics['p_value'] = p_val
        
        return metrics


if __name__ == '__main__':
    print("AI Productivity Analyzer Example")
    print("=" * 60)
    
    # Sample data
    df = pd.DataFrame({
        'ticker': ['PLD'] * 100,
        'date': pd.date_range('2022-01-01', periods=100, freq='D'),
        'ai_mentioned': np.random.choice([True, False], 100, p=[0.15, 0.85]),
        'sentiment_compound': np.random.normal(0.2, 0.3, 100)
    })
    
    analyzer = AIProductivityAnalyzer()
    
    # Classify adopters
    adopters = analyzer.classify_ai_adopters(df)
    print("\nAI Adopter Classification:")
    print(adopters)
    
    # Measure change
    metrics = analyzer.measure_productivity_change(df, '2022-11-30')
    print("\nProductivity Change:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")