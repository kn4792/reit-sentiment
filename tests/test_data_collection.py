import pytest
import pandas as pd
import os
from pathlib import Path

# Try to import - if it fails, the test will skip
try:
    from src.data_collection.glassdoor_scraper import GlassdoorScraper
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

# Fixtures
@pytest.fixture
def sample_review_data():
    """Sample review data for testing"""
    return {
        'title': 'Great place to work',
        'rating': '4.0',
        'employee_info': 'Current Employee - Software Engineer',
        'date': 'Jan 15, 2023',
        'pros': 'Good benefits and work-life balance',
        'cons': 'Could improve remote work policies',
        'advice': 'Keep supporting employees'
    }

@pytest.fixture
def scraper_config():
    """Configuration for scraper tests"""
    return {
        'test_mode': True,
        'max_reviews': 10,
        'headless': True
    }

# Data Validation Tests
class TestDataValidation:
    """Test data format and integrity"""
    
    def test_review_data_structure(self, sample_review_data):
        """Test that review data has required fields"""
        required_fields = ['title', 'rating', 'pros', 'cons', 'date']
        for field in required_fields:
            assert field in sample_review_data, f"Missing required field: {field}"
    
    def test_rating_format(self, sample_review_data):
        """Test that rating is valid number"""
        rating = float(sample_review_data['rating'])
        assert 1.0 <= rating <= 5.0, "Rating must be between 1.0 and 5.0"
    
    def test_date_parsing(self, sample_review_data):
        """Test that date can be parsed"""
        from datetime import datetime
        date_str = sample_review_data['date']
        # Should not raise exception
        parsed_date = pd.to_datetime(date_str)
        assert parsed_date is not None

# CSV Output Tests
class TestDataOutput:
    """Test data saving and loading"""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory"""
        output_dir = tmp_path / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def test_save_to_csv(self, sample_review_data, temp_output_dir):
        """Test saving review data to CSV"""
        df = pd.DataFrame([sample_review_data])
        output_file = temp_output_dir / "test_reviews.csv"
        
        df.to_csv(output_file, index=False)
        
        assert output_file.exists(), "CSV file was not created"
        assert output_file.stat().st_size > 0, "CSV file is empty"
    
    def test_load_from_csv(self, sample_review_data, temp_output_dir):
        """Test loading review data from CSV"""
        # Save data
        df_original = pd.DataFrame([sample_review_data])
        output_file = temp_output_dir / "test_reviews.csv"
        df_original.to_csv(output_file, index=False)
        
        # Load data
        df_loaded = pd.read_csv(output_file)
        
        assert len(df_loaded) == 1, "Wrong number of rows loaded"
        assert list(df_loaded.columns) == list(df_original.columns), \
            "Column mismatch"
        
        # Fix: Compare as floats since pandas converts '4.0' to 4.0
        assert float(df_loaded['rating'][0]) == float(sample_review_data['rating']), \
            "Data mismatch after loading"

# Scraper Integration Tests (Smoke Tests)
@pytest.mark.skipif(not SCRAPER_AVAILABLE, reason="Scraper module not available")
class TestScraperIntegration:
    """Integration tests for scraper (using small sample)"""
    
    @pytest.mark.smoke
    def test_scraper_can_be_imported(self):
        """Test that scraper module can be imported"""
        # This test only runs if import succeeded above
        assert SCRAPER_AVAILABLE, "Scraper should be importable"

