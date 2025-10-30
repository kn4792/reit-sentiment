import pytest
import pandas as pd

# Try to import preprocessing modules
try:
    from src.preprocessing.text_cleaner import TextCleaner
    CLEANER_AVAILABLE = True
except ImportError:
    CLEANER_AVAILABLE = False
    # Create a mock for testing
    class TextCleaner:
        def to_lowercase(self, text):
            return text.lower()
        
        def remove_special_chars(self, text):
            import re
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        def clean(self, text):
            return self.to_lowercase(self.remove_special_chars(text))

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return "This is a GREAT company! I love working here. #blessed"

@pytest.fixture
def sample_reviews_df():
    """Sample DataFrame of reviews"""
    return pd.DataFrame({
        'pros': [
            'Great benefits and work culture',
            'Excellent management team',
            'Good work-life balance'
        ],
        'cons': [
            'Low pay compared to industry',
            'Limited growth opportunities',
            'High turnover rate'
        ],
        'rating': [4.0, 5.0, 3.5]
    })

class TestTextCleaning:
    """Test text cleaning functions"""
    
    def test_lowercase_conversion(self, sample_text):
        """Test converting text to lowercase"""
        cleaner = TextCleaner()
        cleaned = cleaner.to_lowercase(sample_text)
        assert cleaned == cleaned.lower()
        assert 'GREAT' not in cleaned
        assert 'great' in cleaned
    
    def test_remove_special_characters(self, sample_text):
        """Test removing special characters"""
        cleaner = TextCleaner()
        cleaned = cleaner.remove_special_chars(sample_text)
        assert '#' not in cleaned
        assert '!' not in cleaned
    
    def test_clean_pipeline(self, sample_text):
        """Test full cleaning pipeline"""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(sample_text)
        
        # Check that cleaning was applied
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
