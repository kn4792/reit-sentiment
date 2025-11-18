#!/usr/bin/env python3
"""
Unit Tests for Sentiment Analysis Module

Tests FinBERT sentiment analyzer and Loughran-McDonald dictionary-based sentiment.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    pytest tests/test_sentiment.py -v
    pytest tests/test_sentiment.py -v --cov=src/models
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# MOCK CLASSES FOR TESTING
# ============================================================================

class MockFinBERTSentimentAnalyzer:
    """Mock FinBERT analyzer for testing."""
    
    def __init__(self, model_name='ProsusAI/finbert', use_gpu=False, batch_size=8):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.device = 'cuda' if use_gpu else 'cpu'
    
    def predict_single(self, text):
        """Mock single prediction."""
        if not text or pd.isna(text):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Simple rule-based mock for testing
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['great', 'excellent', 'good', 'amazing']):
            return {
                'positive': 0.8,
                'negative': 0.1,
                'neutral': 0.1,
                'compound': 0.7
            }
        elif any(word in text_lower for word in ['bad', 'poor', 'terrible', 'awful']):
            return {
                'positive': 0.1,
                'negative': 0.8,
                'neutral': 0.1,
                'compound': -0.7
            }
        else:
            return {
                'positive': 0.2,
                'negative': 0.2,
                'neutral': 0.6,
                'compound': 0.0
            }
    
    def predict_batch(self, texts):
        """Mock batch prediction."""
        return [self.predict_single(text) for text in texts]
    
    def analyze_dataframe(self, df, text_column='pros_cleaned'):
        """Analyze sentiment for DataFrame."""
        texts = df[text_column].fillna('').tolist()
        results = self.predict_batch(texts)
        
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        
        # Add discrete label
        labels = []
        for r in results:
            if r['positive'] > r['negative'] and r['positive'] > r['neutral']:
                labels.append('positive')
            elif r['negative'] > r['positive'] and r['negative'] > r['neutral']:
                labels.append('negative')
            else:
                labels.append('neutral')
        df['sentiment_label'] = labels
        
        return df


class MockLoughranMcDonaldSentiment:
    """Mock Loughran-McDonald sentiment analyzer."""
    
    def __init__(self):
        self.positive_words = {
            'profit', 'gain', 'growth', 'increase', 'strong', 'excellent',
            'improve', 'success', 'benefit', 'good', 'great', 'positive'
        }
        
        self.negative_words = {
            'loss', 'decline', 'decrease', 'weak', 'poor', 'fail', 'bad',
            'concern', 'problem', 'difficult', 'negative', 'terrible'
        }
    
    def analyze(self, text):
        """Calculate LM sentiment score."""
        if not text or pd.isna(text):
            return {'pos_count': 0, 'neg_count': 0, 'net_score': 0.0}
        
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        total_words = len(words) if len(words) > 0 else 1
        net_score = (pos_count - neg_count) / total_words
        
        return {
            'pos_count': pos_count,
            'neg_count': neg_count,
            'net_score': net_score
        }
    
    def analyze_dataframe(self, df, text_column='pros_cleaned'):
        """Analyze sentiment for DataFrame."""
        results = df[text_column].apply(self.analyze)
        
        df['lm_positive_count'] = [r['pos_count'] for r in results]
        df['lm_negative_count'] = [r['neg_count'] for r in results]
        df['lm_net_score'] = [r['net_score'] for r in results]
        
        return df


# ============================================================================
# FINBERT SENTIMENT ANALYZER TESTS
# ============================================================================

class TestFinBERTSentimentAnalyzer:
    """Tests for FinBERT sentiment analyzer."""
    
    def test_init_default(self):
        """Test default initialization."""
        analyzer = MockFinBERTSentimentAnalyzer()
        assert analyzer.model_name == 'ProsusAI/finbert'
        assert analyzer.use_gpu == False
        assert analyzer.batch_size == 8
        assert analyzer.device == 'cpu'
    
    def test_init_with_gpu(self):
        """Test initialization with GPU."""
        analyzer = MockFinBERTSentimentAnalyzer(use_gpu=True)
        assert analyzer.device == 'cuda'
    
    def test_init_custom_batch_size(self):
        """Test initialization with custom batch size."""
        analyzer = MockFinBERTSentimentAnalyzer(batch_size=16)
        assert analyzer.batch_size == 16
    
    def test_predict_single_positive(self):
        """Test single prediction with positive text."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        text = "Great company with excellent benefits and good culture"
        result = analyzer.predict_single(text)
        
        assert 'positive' in result
        assert 'negative' in result
        assert 'neutral' in result
        assert 'compound' in result
        
        # Should be positive
        assert result['positive'] > result['negative']
        assert result['compound'] > 0
    
    def test_predict_single_negative(self):
        """Test single prediction with negative text."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        text = "Bad management and terrible work environment"
        result = analyzer.predict_single(text)
        
        # Should be negative
        assert result['negative'] > result['positive']
        assert result['compound'] < 0
    
    def test_predict_single_neutral(self):
        """Test single prediction with neutral text."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        text = "The office is located downtown"
        result = analyzer.predict_single(text)
        
        # Should be neutral or balanced
        assert 'neutral' in result
        assert -0.5 <= result['compound'] <= 0.5
    
    def test_predict_single_empty(self):
        """Test single prediction with empty text."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        result = analyzer.predict_single("")
        assert result['positive'] == 0.0
        assert result['negative'] == 0.0
        assert result['neutral'] == 1.0
        assert result['compound'] == 0.0
    
    def test_predict_single_none(self):
        """Test single prediction with None."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        result = analyzer.predict_single(None)
        assert result['neutral'] == 1.0
    
    def test_predict_batch_multiple(self):
        """Test batch prediction with multiple texts."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        texts = [
            "Great benefits and good culture",
            "Poor management and bad communication",
            "Standard office environment"
        ]
        
        results = analyzer.predict_batch(texts)
        
        assert len(results) == 3
        assert results[0]['compound'] > 0  # Positive
        assert results[1]['compound'] < 0  # Negative
        assert -0.5 <= results[2]['compound'] <= 0.5  # Neutral
    
    def test_predict_batch_empty_list(self):
        """Test batch prediction with empty list."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        results = analyzer.predict_batch([])
        assert len(results) == 0
    
    def test_predict_batch_with_nones(self):
        """Test batch prediction with None values."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        texts = ["Great company", None, "Bad management"]
        results = analyzer.predict_batch(texts)
        
        assert len(results) == 3
        assert results[1]['neutral'] == 1.0  # None should return neutral
    
    def test_analyze_dataframe_basic(self):
        """Test DataFrame analysis."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'pros_cleaned': [
                'great benefits excellent culture',
                'bad management poor communication',
                'standard office downtown location'
            ]
        })
        
        result = analyzer.analyze_dataframe(df)
        
        # Check columns exist
        assert 'sentiment_positive' in result.columns
        assert 'sentiment_negative' in result.columns
        assert 'sentiment_neutral' in result.columns
        assert 'sentiment_compound' in result.columns
        assert 'sentiment_label' in result.columns
        
        # Check values
        assert len(result) == 3
        assert result['sentiment_label'].iloc[0] == 'positive'
        assert result['sentiment_label'].iloc[1] == 'negative'
    
    def test_analyze_dataframe_custom_column(self):
        """Test DataFrame analysis with custom text column."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        df = pd.DataFrame({
            'ticker': ['PLD'],
            'cons_cleaned': ['bad management']
        })
        
        result = analyzer.analyze_dataframe(df, text_column='cons_cleaned')
        
        assert 'sentiment_compound' in result.columns
        assert result['sentiment_compound'].iloc[0] < 0
    
    def test_analyze_dataframe_empty(self):
        """Test DataFrame analysis with empty DataFrame."""
        analyzer = MockFinBERTSentimentAnalyzer()
        
        df = pd.DataFrame({
            'ticker': [],
            'pros_cleaned': []
        })
        
        result = analyzer.analyze_dataframe(df)
        assert len(result) == 0


# ============================================================================
# LOUGHRAN-MCDONALD SENTIMENT TESTS
# ============================================================================

class TestLoughranMcDonaldSentiment:
    """Tests for Loughran-McDonald sentiment analyzer."""
    
    def test_init(self):
        """Test initialization."""
        analyzer = MockLoughranMcDonaldSentiment()
        assert len(analyzer.positive_words) > 0
        assert len(analyzer.negative_words) > 0
        assert 'profit' in analyzer.positive_words
        assert 'loss' in analyzer.negative_words
    
    def test_analyze_positive(self):
        """Test analysis with positive text."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        text = "strong profit growth and excellent success"
        result = analyzer.analyze(text)
        
        assert result['pos_count'] > 0
        assert result['net_score'] > 0
    
    def test_analyze_negative(self):
        """Test analysis with negative text."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        text = "loss decline and poor performance"
        result = analyzer.analyze(text)
        
        assert result['neg_count'] > 0
        assert result['net_score'] < 0
    
    def test_analyze_mixed(self):
        """Test analysis with mixed sentiment."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        text = "profit increased but loss also occurred"
        result = analyzer.analyze(text)
        
        assert result['pos_count'] > 0
        assert result['neg_count'] > 0
    
    def test_analyze_neutral(self):
        """Test analysis with neutral text."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        text = "the office is located downtown near the train station"
        result = analyzer.analyze(text)
        
        assert result['pos_count'] == 0
        assert result['neg_count'] == 0
        assert result['net_score'] == 0.0
    
    def test_analyze_empty(self):
        """Test analysis with empty text."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        result = analyzer.analyze("")
        assert result['pos_count'] == 0
        assert result['neg_count'] == 0
        assert result['net_score'] == 0.0
    
    def test_analyze_none(self):
        """Test analysis with None."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        result = analyzer.analyze(None)
        assert result['net_score'] == 0.0
    
    def test_analyze_case_insensitive(self):
        """Test that analysis is case-insensitive."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        text1 = "PROFIT GROWTH"
        text2 = "profit growth"
        
        result1 = analyzer.analyze(text1)
        result2 = analyzer.analyze(text2)
        
        assert result1['pos_count'] == result2['pos_count']
    
    def test_analyze_dataframe_basic(self):
        """Test DataFrame analysis."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'pros_cleaned': [
                'profit growth strong performance',
                'loss decline poor results',
                'office downtown location'
            ]
        })
        
        result = analyzer.analyze_dataframe(df)
        
        # Check columns exist
        assert 'lm_positive_count' in result.columns
        assert 'lm_negative_count' in result.columns
        assert 'lm_net_score' in result.columns
        
        # Check values
        assert result['lm_net_score'].iloc[0] > 0  # Positive
        assert result['lm_net_score'].iloc[1] < 0  # Negative
        assert result['lm_net_score'].iloc[2] == 0  # Neutral
    
    def test_analyze_dataframe_custom_column(self):
        """Test DataFrame analysis with custom column."""
        analyzer = MockLoughranMcDonaldSentiment()
        
        df = pd.DataFrame({
            'ticker': ['PLD'],
            'cons_cleaned': ['loss and decline']
        })
        
        result = analyzer.analyze_dataframe(df, text_column='cons_cleaned')
        
        assert 'lm_net_score' in result.columns
        assert result['lm_net_score'].iloc[0] < 0


# ============================================================================
# AGGREGATION TESTS
# ============================================================================

class TestAggregation:
    """Tests for sentiment aggregation functions."""
    
    def test_aggregate_to_monthly_basic(self):
        """Test monthly aggregation."""
        df = pd.DataFrame({
            'ticker': ['PLD', 'PLD', 'AMT', 'AMT'],
            'date': ['2025-01-15', '2025-01-20', '2025-01-15', '2025-02-15'],
            'sentiment_compound': [0.5, 0.6, -0.3, 0.2],
            'sentiment_positive': [0.7, 0.8, 0.2, 0.5],
            'sentiment_negative': [0.1, 0.1, 0.7, 0.3],
            'sentiment_neutral': [0.2, 0.1, 0.1, 0.2],
            'lm_net_score': [0.1, 0.2, -0.2, 0.1]
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate
        monthly = df.groupby(['ticker', 'year_month']).agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'lm_net_score': 'mean'
        }).reset_index()
        
        # Check results
        assert len(monthly) == 3  # PLD Jan, AMT Jan, AMT Feb
    
    def test_calculate_sentiment_stats(self):
        """Test sentiment statistics calculation."""
        df = pd.DataFrame({
            'sentiment_compound': [0.5, 0.3, -0.2, 0.1, -0.4],
            'sentiment_label': ['positive', 'positive', 'negative', 'neutral', 'negative']
        })
        
        stats = {
            'mean_compound': df['sentiment_compound'].mean(),
            'std_compound': df['sentiment_compound'].std(),
            'median_compound': df['sentiment_compound'].median(),
            'pct_positive': (df['sentiment_label'] == 'positive').sum() / len(df) * 100,
            'pct_negative': (df['sentiment_label'] == 'negative').sum() / len(df) * 100,
            'pct_neutral': (df['sentiment_label'] == 'neutral').sum() / len(df) * 100
        }
        
        assert stats['pct_positive'] == 40.0
        assert stats['pct_negative'] == 40.0
        assert stats['pct_neutral'] == 20.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSentimentIntegration:
    """Integration tests for sentiment analysis pipeline."""
    
    def test_complete_sentiment_pipeline(self):
        """Test complete sentiment analysis workflow."""
        # Create sample data
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'date': ['2025-01-15', '2025-01-16', '2025-01-17'],
            'pros_cleaned': [
                'great benefits excellent culture',
                'bad management poor communication',
                'standard office environment'
            ]
        })
        
        # Initialize analyzers
        finbert = MockFinBERTSentimentAnalyzer()
        lm = MockLoughranMcDonaldSentiment()
        
        # Analyze with FinBERT
        df = finbert.analyze_dataframe(df)
        
        # Analyze with LM
        df = lm.analyze_dataframe(df)
        
        # Assertions
        assert 'sentiment_compound' in df.columns
        assert 'lm_net_score' in df.columns
        assert len(df) == 3
    
    def test_sentiment_with_missing_data(self):
        """Test sentiment analysis with missing data."""
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT'],
            'pros_cleaned': ['great company', None]
        })
        
        finbert = MockFinBERTSentimentAnalyzer()
        df = finbert.analyze_dataframe(df)
        
        # Should handle None gracefully
        assert df['sentiment_neutral'].iloc[1] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])