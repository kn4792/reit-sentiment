"""
Integration tests for Phase 2: Model Training & Evaluation

This module tests the complete model training pipeline including:
- Sentiment analysis with FinBERT
- Keyword extraction with TF-IDF
- Model output validation
- Model evaluation metrics

Author: Konain Niaz
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import project modules
from scripts.keyword_extraction import KeywordExtractor
from scripts.sentiment_analysis import FinBERTSentimentAnalyzer

@pytest.mark.integration
class TestModelTrainingPipeline:
    """Integration tests for complete model training pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment with sample data"""
        self.results_dir = tmp_path / "results"
        self.results_dir.mkdir()
        
        # Create sample training data
        self.train_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'rating': np.random.uniform(2.0, 5.0, 20),
            'pros': [
                'Great company culture and team',
                'Excellent benefits and compensation',
                'Good work-life balance',
                'Strong leadership and management',
                'Innovative AI technology',
                'Supportive colleagues',
                'Career growth opportunities',
                'Flexible work arrangements',
                'Competitive salary',
                'Modern office space',
                'Great REIT performance',
                'Stable company',
                'Good training programs',
                'Nice work environment',
                'Strong market position',
                'Excellent management team',
                'Good company values',
                'Supportive culture',
                'Great benefits package',
                'Good career progression'
            ],
            'cons': [
                'Long working hours',
                'Limited remote work',
                'Slow promotion process',
                'Outdated technology',
                'Poor communication',
                'High pressure environment',
                'Lack of flexibility',
                'Low pay for junior roles',
                'Micromanagement issues',
                'Limited learning opportunities',
                'Old infrastructure',
                'Bureaucratic processes',
                'Unclear expectations',
                'Too many meetings',
                'Limited autonomy',
                'Poor work-life balance',
                'Insufficient resources',
                'Lack of recognition',
                'Slow decision making',
                'Limited innovation'
            ],
            'company': ['TestREIT'] * 20,
            'ticker': ['TEST'] * 20
        })
        
        # Create validation and test data (smaller)
        self.val_data = self.train_data.iloc[:5].copy()
        self.test_data = self.train_data.iloc[5:10].copy()
        
        # Save data
        self.train_file = self.results_dir / "train.csv"
        self.val_file = self.results_dir / "val.csv"
        self.test_file = self.results_dir / "test.csv"
        
        self.train_data.to_csv(self.train_file, index=False)
        self.val_data.to_csv(self.val_file, index=False)
        self.test_data.to_csv(self.test_file, index=False)
        
        yield
        
    def test_training_data_loaded(self):
        """Test that training data is loaded correctly"""
        train_df = pd.read_csv(self.train_file)
        assert len(train_df) > 0
        assert 'pros' in train_df.columns
        assert 'cons' in train_df.columns
        
    def test_sentiment_analyzer_initialization(self):
        """Test FinBERTSentimentAnalyzer initialization"""
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        assert analyzer is not None
        assert hasattr(analyzer, 'predict_single')
        
    def test_sentiment_analysis_on_sample(self):
        """Test sentiment analysis on sample text"""
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        pos_text = "Great company culture and amazing team"
        pos_score = analyzer.predict_single(pos_text)['compound']
        assert isinstance(pos_score, (float, np.floating))
        assert -1 <= pos_score <= 1
        neg_text = "Terrible management and poor work environment"
        neg_score = analyzer.predict_single(neg_text)['compound']
        assert isinstance(neg_score, (float, np.floating))
        assert neg_score < pos_score
        
    def test_sentiment_analysis_batch_processing(self):
        """Test sentiment analysis on batch of texts"""
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        train_df = pd.read_csv(self.train_file)
        sentiments = []
        for text in train_df['pros'].head(5):
            score = analyzer.predict_single(text)['compound']
            sentiments.append(score)
        assert len(sentiments) == 5
        assert all(isinstance(s, (float, np.floating)) for s in sentiments)
        
    def test_sentiment_scores_in_valid_range(self):
        """Test that all sentiment scores are in valid range"""
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        train_df = pd.read_csv(self.train_file)
        train_df['sentiment'] = train_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        assert train_df['sentiment'].min() >= -1
        assert train_df['sentiment'].max() <= 1
        assert train_df['sentiment'].notna().all()
        
    def test_keyword_extractor_initialization(self):
        """Test KeywordExtractor initialization"""
        extractor = KeywordExtractor(max_features=100, ngram_range=(1,2), min_df=2, max_df=0.8)
        assert extractor is not None
        assert hasattr(extractor, 'fit')
        assert hasattr(extractor, 'extract_top_keywords')
        
    def test_keyword_extraction_fit(self):
        """Test keyword extraction fitting"""
        train_df = pd.read_csv(self.train_file)
        extractor = KeywordExtractor(max_features=50, ngram_range=(1,2), min_df=2, max_df=0.8)
        extractor.fit(train_df['pros'])
        keywords = extractor.extract_top_keywords(train_df['pros'], top_k=10)
        assert len(keywords) > 0
        assert len(keywords) <= 10
        
    def test_keyword_extraction_output_format(self):
        """Test keyword extraction output format"""
        train_df = pd.read_csv(self.train_file)
        extractor = KeywordExtractor(max_features=50, ngram_range=(1,2), min_df=2, max_df=0.8)
        extractor.fit(train_df['pros'])
        keywords = extractor.extract_top_keywords(train_df['pros'], top_k=10)
        assert isinstance(keywords, list)
        for item in keywords:
            assert isinstance(item, tuple), "Each keyword should be tuple (word, score) format"
        
    def test_model_output_file_creation(self):
        """Test that model creates output files"""
        train_df = pd.read_csv(self.train_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        train_df['sentiment_score'] = train_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        output_file = self.results_dir / "sentiments_train.csv"
        train_df.to_csv(output_file, index=False)
        assert output_file.exists()
        result_df = pd.read_csv(output_file)
        assert 'sentiment_score' in result_df.columns
        assert len(result_df) == len(train_df)
        
    def test_model_output_has_required_columns(self):
        """Test that model output has all required columns"""
        train_df = pd.read_csv(self.train_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        train_df['sentiment_score'] = train_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        train_df['sentiment_label'] = train_df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
        )
        required_cols = ['date', 'rating', 'pros', 'cons', 'sentiment_score', 'sentiment_label']
        for col in required_cols:
            assert col in train_df.columns
        
    def test_sentiment_distribution(self):
        """Test that sentiment scores have reasonable distribution"""
        train_df = pd.read_csv(self.train_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        train_df['sentiment'] = train_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        mean_sentiment = train_df['sentiment'].mean()
        std_sentiment = train_df['sentiment'].std()
        assert mean_sentiment > -0.5
        assert std_sentiment > 0
        
    def test_model_evaluation_metrics_calculation(self):
        """Test calculation of evaluation metrics"""
        predictions = np.array([0.8, 0.6, -0.3, 0.1, -0.5])
        actuals = np.array([1, 1, -1, 1, -1])
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        assert isinstance(correlation, (float, np.floating))
        assert -1 <= correlation <= 1
        
    def test_model_reproducibility(self):
        """Test that model produces consistent results with same input"""
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        text = "Great company culture and team"
        score1 = analyzer.predict_single(text)['compound']
        score2 = analyzer.predict_single(text)['compound']
        assert abs(score1 - score2) < 1e-5
        
    def test_validation_set_evaluation(self):
        """Test model evaluation on validation set"""
        val_df = pd.read_csv(self.val_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        val_df['sentiment'] = val_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        assert len(val_df) > 0
        assert val_df['sentiment'].notna().all()
        
    def test_test_set_evaluation(self):
        """Test model evaluation on test set"""
        test_df = pd.read_csv(self.test_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        test_df['sentiment'] = test_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        assert len(test_df) > 0
        assert test_df['sentiment'].notna().all()
        
    def test_complete_training_pipeline(self):
        """Test complete model training pipeline end-to-end"""
        train_df = pd.read_csv(self.train_file)
        val_df = pd.read_csv(self.val_file)
        test_df = pd.read_csv(self.test_file)
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        extractor = KeywordExtractor(max_features=50, ngram_range=(1,2), min_df=2, max_df=0.8)

        train_df['sentiment_score'] = train_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        val_df['sentiment_score'] = val_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])
        test_df['sentiment_score'] = test_df['pros'].apply(lambda t: analyzer.predict_single(t)['compound'])

        extractor.fit(train_df['pros'])
        keywords = extractor.extract_top_keywords(train_df['pros'], top_k=10)

        train_output = self.results_dir / "sentiments_train.csv"
        val_output = self.results_dir / "sentiments_val.csv"
        test_output = self.results_dir / "sentiments_test.csv"
        keywords_output = self.results_dir / "keywords_train.csv"

        train_df.to_csv(train_output, index=False)
        val_df.to_csv(val_output, index=False)
        test_df.to_csv(test_output, index=False)
        pd.DataFrame({'keyword': [word for word, _ in keywords]}).to_csv(keywords_output, index=False)

        assert train_output.exists()
        assert val_output.exists()
        assert test_output.exists()
        assert keywords_output.exists()

        train_mean_sentiment = train_df['sentiment_score'].mean()
        val_mean_sentiment = val_df['sentiment_score'].mean()
        assert -1 <= train_mean_sentiment <= 1
        assert -1 <= val_mean_sentiment <= 1

@pytest.mark.unit
class TestSentimentAnalysis:
    """Unit tests for sentiment analysis"""
    
    def test_positive_sentiment_detection(self):
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        positive_texts = [
            "Excellent company with great benefits",
            "Amazing team and wonderful culture",
            "Best place to work ever"
        ]
        for text in positive_texts:
            score = analyzer.predict_single(text)['compound']
            assert score > 0
    
    def test_negative_sentiment_detection(self):
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        negative_texts = [
            "Terrible management and poor culture",
            "Worst company ever with bad benefits",
            "Horrible work environment"
        ]
        for text in negative_texts:
            score = analyzer.predict_single(text)['compound']
            assert score < 0
    
    def test_neutral_sentiment_detection(self):
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        neutral_text = "The company is located in New York"
        score = analyzer.predict_single(neutral_text)['compound']
        assert -0.5 < score < 0.5
    
    def test_empty_text_handling(self):
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        score = analyzer.predict_single("")['compound']
        assert isinstance(score, (float, np.floating))
    
    def test_long_text_handling(self):
        analyzer = FinBERTSentimentAnalyzer(model_name='ProsusAI/finbert', use_gpu=False, batch_size=8)
        long_text = "Great company " * 100
        score = analyzer.predict_single(long_text)['compound']
        assert isinstance(score, (float, np.floating))
        assert -1 <= score <= 1

@pytest.mark.unit
class TestKeywordExtraction:
    """Unit tests for keyword extraction"""
    
    def test_tfidf_keyword_extraction(self):
        texts = [
            "Great company culture and team",
            "Excellent benefits and compensation",
            "Good work-life balance and culture"
        ]
        extractor = KeywordExtractor(max_features=20, ngram_range=(1,2), min_df=2, max_df=0.8)
        extractor.fit(texts)
        keywords = extractor.extract_top_keywords(texts, top_k=5)
        assert len(keywords) > 0
        assert len(keywords) <= 5

    def test_empty_corpus_handling(self):
        extractor = KeywordExtractor(max_features=20, ngram_range=(1,2), min_df=2, max_df=0.8)
        with pytest.raises(Exception):
            extractor.fit([])

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
