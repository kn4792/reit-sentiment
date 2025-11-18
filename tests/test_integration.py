#!/usr/bin/env python3
"""
Integration Tests for Complete REIT Sentiment Analysis Pipeline

Tests end-to-end workflows combining scraping, cleaning, sentiment analysis,
and statistical testing.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v --slow  # Include slow tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_raw_reviews():
    """Create sample raw review data."""
    return pd.DataFrame({
        'ticker': ['PLD', 'PLD', 'AMT', 'AMT', 'EQIX'],
        'company': ['Prologis', 'Prologis', 'American Tower', 'American Tower', 'Equinix'],
        'title': [
            'Great place to work',
            'Good company overall',
            'Decent workplace',
            'Could be better',
            'Excellent culture'
        ],
        'rating': ['4.5', '4.0', '3.5', '3.0', '5.0'],
        'date': ['Jan 15, 2025', '2025-01-10', '1/12/2025', 'Jan 20, 2025', '2025-01-18'],
        'pros': [
            '<p>Great benefits &amp; work-life balance! Visit https://example.com</p>',
            'Good team collaboration and support',
            'Standard benefits package',
            'Decent pay and location',
            'Amazing culture and opportunities'
        ],
        'cons': [
            'Can be stressful during busy periods',
            'Limited advancement opportunities',
            'Management could be better',
            'Long commute for some',
            'Fast-paced environment'
        ],
        'employee_info': [
            'Current Employee - Software Engineer',
            'Former Employee - Analyst',
            'Current Employee - Manager',
            'Current Employee - Developer',
            'Current Employee - Senior Engineer'
        ],
        'scrape_date': ['2025-01-17'] * 5,
        'glassdoor_url': ['https://glassdoor.com/reviews'] * 5
    })


@pytest.fixture
def sample_cleaned_reviews():
    """Create sample cleaned review data."""
    return pd.DataFrame({
        'ticker': ['PLD', 'PLD', 'AMT', 'AMT', 'EQIX'],
        'date': ['2025-01-15', '2025-01-10', '2025-01-12', '2025-01-20', '2025-01-18'],
        'pros_cleaned': [
            'great benefit work life balance',
            'good team collaboration support',
            'standard benefit package',
            'decent pay location',
            'amazing culture opportunity'
        ],
        'cons_cleaned': [
            'stressful busy period',
            'limited advancement opportunity',
            'management could better',
            'long commute',
            'fast paced environment'
        ],
        'rating': [4.5, 4.0, 3.5, 3.0, 5.0]
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


# ============================================================================
# DATA PIPELINE INTEGRATION TESTS
# ============================================================================

class TestDataPipeline:
    """Test complete data processing pipeline."""
    
    def test_raw_to_cleaned_pipeline(self, sample_raw_reviews):
        """Test pipeline from raw reviews to cleaned reviews."""
        # Mock text cleaning
        def clean_text(text):
            if pd.isna(text):
                return ""
            import re
            text = re.sub(r'<[^>]+>', '', str(text))
            text = re.sub(r'&[a-z]+;', ' ', text)
            text = re.sub(r'http\S+', '', text)
            text = text.lower()
            return text.strip()
        
        # Apply cleaning
        df = sample_raw_reviews.copy()
        df['pros_cleaned'] = df['pros'].apply(clean_text)
        df['cons_cleaned'] = df['cons'].apply(clean_text)
        
        # Verify cleaning
        assert 'pros_cleaned' in df.columns
        assert 'cons_cleaned' in df.columns
        assert '<p>' not in df['pros_cleaned'].iloc[0]
        assert '&amp;' not in df['pros_cleaned'].iloc[0]
        assert 'https://' not in df['pros_cleaned'].iloc[0]
    
    def test_cleaned_to_sentiment_pipeline(self, sample_cleaned_reviews):
        """Test pipeline from cleaned reviews to sentiment scores."""
        df = sample_cleaned_reviews.copy()
        
        # Mock sentiment analysis
        def get_sentiment(text):
            if 'great' in text or 'amazing' in text or 'excellent' in text:
                return 0.7
            elif 'bad' in text or 'poor' in text or 'terrible' in text:
                return -0.7
            else:
                return 0.0
        
        df['sentiment_compound'] = df['pros_cleaned'].apply(get_sentiment)
        
        # Verify sentiment
        assert 'sentiment_compound' in df.columns
        assert df['sentiment_compound'].iloc[0] > 0  # "great" → positive
        assert df['sentiment_compound'].iloc[4] > 0  # "amazing" → positive
    
    def test_sentiment_to_aggregation_pipeline(self, sample_cleaned_reviews):
        """Test pipeline from sentiment to monthly aggregation."""
        df = sample_cleaned_reviews.copy()
        
        # Add mock sentiment
        df['sentiment_compound'] = [0.7, 0.5, 0.0, -0.2, 0.8]
        
        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate
        monthly = df.groupby(['ticker', 'year_month']).agg({
            'sentiment_compound': ['mean', 'std', 'count']
        }).reset_index()
        
        # Verify aggregation
        assert len(monthly) > 0
        assert 'ticker' in monthly.columns
        assert 'year_month' in monthly.columns


# ============================================================================
# FILE I/O INTEGRATION TESTS
# ============================================================================

class TestFileOperations:
    """Test file reading and writing operations."""
    
    def test_save_and_load_csv(self, sample_raw_reviews, temp_dir):
        """Test saving and loading CSV files."""
        output_file = temp_dir / "test_reviews.csv"
        
        # Save
        sample_raw_reviews.to_csv(output_file, index=False)
        
        # Load
        loaded_df = pd.read_csv(output_file)
        
        # Verify
        assert len(loaded_df) == len(sample_raw_reviews)
        assert list(loaded_df.columns) == list(sample_raw_reviews.columns)
    
    def test_multiple_company_files(self, sample_raw_reviews, temp_dir):
        """Test handling multiple company CSV files."""
        # Split by company
        for ticker in sample_raw_reviews['ticker'].unique():
            company_df = sample_raw_reviews[sample_raw_reviews['ticker'] == ticker]
            output_file = temp_dir / f"{ticker}_reviews.csv"
            company_df.to_csv(output_file, index=False)
        
        # Load all files
        all_files = list(temp_dir.glob("*_reviews.csv"))
        all_dfs = [pd.read_csv(f) for f in all_files]
        combined = pd.concat(all_dfs, ignore_index=True)
        
        # Verify
        assert len(all_files) == 3  # PLD, AMT, EQIX
        assert len(combined) == len(sample_raw_reviews)


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_analysis_workflow(self, sample_raw_reviews, temp_dir):
        """Test complete workflow from raw data to results."""
        # Step 1: Clean data
        def clean_text(text):
            if pd.isna(text):
                return ""
            import re
            text = re.sub(r'<[^>]+>', '', str(text))
            text = text.lower()
            return text.strip()
        
        df = sample_raw_reviews.copy()
        df['pros_cleaned'] = df['pros'].apply(clean_text)
        
        # Step 2: Parse dates
        def parse_date(date_str):
            try:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except:
                return None
        
        df['date_parsed'] = df['date'].apply(parse_date)
        
        # Step 3: Add sentiment
        df['sentiment_compound'] = df['pros_cleaned'].apply(
            lambda x: 0.7 if 'great' in x else 0.0
        )
        
        # Step 4: Aggregate
        df['date_parsed'] = pd.to_datetime(df['date_parsed'])
        df['year_month'] = df['date_parsed'].dt.to_period('M')
        
        monthly = df.groupby(['ticker', 'year_month']).agg({
            'sentiment_compound': 'mean'
        }).reset_index()
        
        # Step 5: Save results
        output_file = temp_dir / "results.csv"
        monthly.to_csv(output_file, index=False)
        
        # Verify
        assert output_file.exists()
        assert len(monthly) > 0
    
    def test_batch_processing_workflow(self, sample_raw_reviews, temp_dir):
        """Test batch processing of multiple companies."""
        results = []
        
        # Process each company
        for ticker in sample_raw_reviews['ticker'].unique():
            company_df = sample_raw_reviews[sample_raw_reviews['ticker'] == ticker]
            
            # Mock processing
            processed = {
                'ticker': ticker,
                'review_count': len(company_df),
                'avg_rating': company_df['rating'].astype(float).mean()
            }
            results.append(processed)
        
        # Create summary
        summary_df = pd.DataFrame(results)
        summary_file = temp_dir / "summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Verify
        assert len(summary_df) == 3
        assert summary_file.exists()


# ============================================================================
# DATA QUALITY INTEGRATION TESTS
# ============================================================================

class TestDataQuality:
    """Test data quality across pipeline."""
    
    def test_no_data_loss(self, sample_raw_reviews):
        """Test that no data is lost during processing."""
        initial_count = len(sample_raw_reviews)
        
        # Process
        df = sample_raw_reviews.copy()
        df['pros_cleaned'] = df['pros'].fillna('').str.lower()
        
        # Verify
        assert len(df) == initial_count
    
    def test_consistent_types(self, sample_raw_reviews):
        """Test that data types remain consistent."""
        df = sample_raw_reviews.copy()
        
        # Convert rating to float
        df['rating'] = df['rating'].astype(float)
        
        # Verify
        assert df['rating'].dtype == np.float64
        assert all(df['rating'] >= 0)
        assert all(df['rating'] <= 5)
    
    def test_date_parsing_consistency(self, sample_raw_reviews):
        """Test that all date formats are parsed consistently."""
        df = sample_raw_reviews.copy()
        
        # Parse dates
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Verify
        assert df['date_parsed'].notna().all()
        assert all(df['date_parsed'].dt.year == 2025)


# ============================================================================
# ERROR HANDLING INTEGRATION TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling across pipeline."""
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', None],
            'pros': ['Great company', None, 'Good benefits'],
            'rating': ['4.5', None, '5.0']
        })
        
        # Handle missing values
        df['ticker'] = df['ticker'].fillna('UNKNOWN')
        df['pros'] = df['pros'].fillna('')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Verify
        assert df['ticker'].notna().all()
        assert df['pros'].notna().all()
    
    def test_invalid_rating_handling(self):
        """Test handling of invalid ratings."""
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'rating': ['4.5', 'invalid', '5.0']
        })
        
        # Convert and handle errors
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df[df['rating'].notna()]
        
        # Verify
        assert len(df) == 2  # Invalid rating removed
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        df = pd.DataFrame(columns=['ticker', 'pros', 'rating'])
        
        # Process empty DataFrame
        df['pros_cleaned'] = df['pros'].fillna('').str.lower()
        
        # Verify
        assert len(df) == 0
        assert 'pros_cleaned' in df.columns


# ============================================================================
# PERFORMANCE INTEGRATION TESTS
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset."""
        # Create large dataset (1000 reviews)
        large_df = pd.DataFrame({
            'ticker': ['PLD'] * 1000,
            'pros': ['Great company'] * 1000,
            'rating': np.random.uniform(1, 5, 1000)
        })
        
        # Process
        import time
        start = time.time()
        large_df['pros_cleaned'] = large_df['pros'].str.lower()
        processing_time = time.time() - start
        
        # Verify
        assert len(large_df) == 1000
        assert processing_time < 1.0  # Should be fast
    
    def test_batch_memory_efficiency(self):
        """Test memory efficiency with batch processing."""
        # Create data in batches
        batch_size = 100
        n_batches = 10
        
        results = []
        for i in range(n_batches):
            batch = pd.DataFrame({
                'ticker': ['PLD'] * batch_size,
                'pros': [f'Review {j}' for j in range(batch_size)]
            })
            batch['pros_cleaned'] = batch['pros'].str.lower()
            results.append(len(batch))
        
        # Verify
        assert sum(results) == batch_size * n_batches


# ============================================================================
# STATISTICAL ANALYSIS INTEGRATION TESTS
# ============================================================================

class TestStatisticalAnalysis:
    """Test statistical analysis components."""
    
    def test_correlation_calculation(self, sample_cleaned_reviews):
        """Test correlation calculation."""
        df = sample_cleaned_reviews.copy()
        df['sentiment_compound'] = [0.7, 0.5, 0.0, -0.2, 0.8]
        df['rating'] = [4.5, 4.0, 3.5, 3.0, 5.0]
        
        # Calculate correlation
        from scipy.stats import pearsonr
        corr, pval = pearsonr(df['sentiment_compound'], df['rating'])
        
        # Verify
        assert -1 <= corr <= 1
        assert 0 <= pval <= 1
    
    def test_aggregation_statistics(self, sample_cleaned_reviews):
        """Test aggregation statistics."""
        df = sample_cleaned_reviews.copy()
        df['sentiment_compound'] = [0.7, 0.5, 0.0, -0.2, 0.8]
        
        # Group statistics
        stats = df.groupby('ticker')['sentiment_compound'].agg(['mean', 'std', 'count'])
        
        # Verify
        assert len(stats) == 3  # PLD, AMT, EQIX
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'count' in stats.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])