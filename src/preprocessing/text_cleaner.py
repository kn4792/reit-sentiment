import re


class TextCleaner:
    """Clean and preprocess text data"""
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        return text.lower()
    
    def remove_special_chars(self, text):
        """Remove special characters"""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def clean(self, text):
        """Full cleaning pipeline"""
        text = self.to_lowercase(text)
        text = self.remove_special_chars(text)
        return text