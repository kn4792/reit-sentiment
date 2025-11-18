"""
Keyword Extraction Module for REIT Reviews

Implements TF-IDF based keyword extraction and category classification.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class KeywordExtractor:
    """
    TF-IDF based keyword extraction for review analysis.
    
    Features:
    - Company-specific keyword importance
    - Multi-word phrases (n-grams)
    - Custom category detection (AI, management, culture, etc.)
    
    Attributes:
        max_features: Maximum number of keywords to extract
        ngram_range: (min_n, max_n) for n-gram extraction
        vectorizer: TF-IDF vectorizer instance
        feature_names: Extracted feature names
    
    Example:
        >>> extractor = KeywordExtractor(max_features=100)
        >>> extractor.fit(texts)
        >>> keywords = extractor.get_top_keywords(n=20)
    """
    
    def __init__(self,
                 max_features: int = 100,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Initialize keyword extractor.
        
        Args:
            max_features: Maximum number of keywords to extract
            ngram_range: (min_n, max_n) for n-gram extraction
            min_df: Minimum document frequency for keywords
            max_df: Maximum document frequency (as fraction)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.feature_names = None
    
    def fit(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer on corpus.
        
        Args:
            texts: List of text documents
            
        Example:
            >>> extractor = KeywordExtractor()
            >>> extractor.fit(df['pros_cleaned'].tolist())
        """
        print(f"🔧 Fitting TF-IDF vectorizer...")
        print(f"  • N-gram range: {self.ngram_range}")
        print(f"  • Max features: {self.max_features}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None  # Already removed in cleaning
        )
        
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✓ Extracted {len(self.feature_names)} features")
    
    def get_top_keywords(self,
                        texts: List[str],
                        top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract top-k keywords from texts.
        
        Args:
            texts: List of text documents
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
            
        Example:
            >>> keywords = extractor.get_top_keywords(texts, top_k=10)
            >>> print(keywords[0])
            ('benefit', 0.543)
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        # Transform texts
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Sum TF-IDF scores across documents
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Get keywords and scores
        keywords = [(self.feature_names[i], scores[i]) for i in top_indices]
        
        return keywords
    
    def extract_by_company(self,
                          df: pd.DataFrame,
                          text_column: str = 'pros_cleaned',
                          top_k: int = 20) -> pd.DataFrame:
        """
        Extract top keywords for each company.
        
        Args:
            df: DataFrame with company reviews
            text_column: Column containing cleaned text
            top_k: Number of keywords per company
            
        Returns:
            DataFrame with columns: ticker, keyword, tfidf_score, rank
            
        Example:
            >>> keywords_df = extractor.extract_by_company(df, top_k=20)
            >>> print(keywords_df.head())
        """
        print(f"\n🔍 Extracting keywords by company...")
        
        results = []
        
        for ticker in df['ticker'].unique():
            company_df = df[df['ticker'] == ticker]
            texts = company_df[text_column].fillna('').tolist()
            
            # Skip if no text
            if not texts or all(t == '' for t in texts):
                continue
            
            # Extract keywords
            keywords = self.get_top_keywords(texts, top_k=top_k)
            
            # Store results
            for rank, (keyword, score) in enumerate(keywords, 1):
                results.append({
                    'ticker': ticker,
                    'keyword': keyword,
                    'tfidf_score': score,
                    'rank': rank
                })
            
            print(f"  ✓ {ticker}: {len(keywords)} keywords")
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Extracted keywords for {results_df['ticker'].nunique()} companies")
        
        return results_df


class CategoryClassifier:
    """
    Classify reviews into predefined categories based on keywords.
    
    Categories:
    - AI/Technology
    - Management
    - Culture
    - Compensation
    - Work-life balance
    
    Attributes:
        categories: Dictionary mapping category names to keyword sets
    
    Example:
        >>> classifier = CategoryClassifier()
        >>> df = classifier.classify_dataframe(df)
        >>> print(df['category_ai_technology'].sum())
        45
    """
    
    def __init__(self):
        """Initialize with predefined category keywords."""
        
        self.categories = {
            'ai_technology': {
                'ai', 'artificial', 'intelligence', 'machine', 'learning',
                'algorithm', 'data', 'analytics', 'automation', 'digital',
                'technology', 'software', 'platform', 'system', 'tool',
                'chatgpt', 'gpt', 'neural', 'model', 'computer', 'vision',
                'nlp', 'deep', 'proptech', 'tech', 'technological'
            },
            
            'management': {
                'management', 'manager', 'leadership', 'executive', 'ceo',
                'director', 'supervisor', 'boss', 'lead', 'senior',
                'decision', 'strategy', 'vision', 'communication',
                'micromanage', 'transparent', 'support', 'feedback'
            },
            
            'culture': {
                'culture', 'team', 'colleague', 'collaborative', 'friendly',
                'environment', 'atmosphere', 'vibe', 'people', 'staff',
                'inclusive', 'diverse', 'diversity', 'respect', 'trust',
                'toxic', 'political', 'clique'
            },
            
            'compensation': {
                'salary', 'pay', 'compensation', 'wage', 'bonus', 'incentive',
                'benefit', 'insurance', 'healthcare', 'retirement', '401k',
                'stock', 'equity', 'raise', 'promotion', 'competitive'
            },
            
            'work_life_balance': {
                'balance', 'flexible', 'flexibility', 'remote', 'wfh',
                'hybrid', 'hour', 'overtime', 'vacation', 'pto', 'leave',
                'weekend', 'schedule', 'burnout', 'stress'
            }
        }
    
    def classify_text(self, text: str) -> Dict[str, bool]:
        """
        Classify single text into categories.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with category names as keys, boolean values
            
        Example:
            >>> classifier = CategoryClassifier()
            >>> result = classifier.classify_text("great benefits and flexible schedule")
            >>> print(result)
            {'compensation': True, 'work_life_balance': True, ...}
        """
        if not text or pd.isna(text):
            return {cat: False for cat in self.categories.keys()}
        
        words = set(text.lower().split())
        
        classifications = {}
        for category, keywords in self.categories.items():
            classifications[category] = bool(words & keywords)
        
        return classifications
    
    def classify_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Classify all reviews in DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with category columns added
            
        Example:
            >>> classifier = CategoryClassifier()
            >>> df = classifier.classify_dataframe(df)
            >>> print(df['category_ai_technology'].sum())
        """
        print(f"\n🏷️  Classifying reviews into categories...")
        
        # Classify each review
        classifications = df[text_column].apply(self.classify_text)
        
        # Add category columns
        df = df.copy()
        for category in self.categories.keys():
            df[f'category_{category}'] = [c[category] for c in classifications]
        
        # Print statistics
        print("\n📊 Category distribution:")
        for category in self.categories.keys():
            count = df[f'category_{category}'].sum()
            pct = (count / len(df)) * 100
            print(f"  • {category}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def get_category_keywords(self, category: str) -> Set[str]:
        """
        Get keywords for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Set of keywords
            
        Example:
            >>> classifier = CategoryClassifier()
            >>> keywords = classifier.get_category_keywords('ai_technology')
            >>> print(len(keywords))
            27
        """
        return self.categories.get(category, set())


class AIAdoptionAnalyzer:
    """
    Analyze AI adoption mentions in reviews over time.
    
    Tracks mentions of AI/technology keywords and calculates
    adoption metrics at company-month level.
    
    Attributes:
        ai_keywords: Set of AI-related keywords
    
    Example:
        >>> ai_analyzer = AIAdoptionAnalyzer()
        >>> df = ai_analyzer.analyze_dataframe(df)
        >>> print(df['ai_mentioned'].sum())
        127
    """
    
    def __init__(self):
        """Initialize with AI-related keywords."""
        
        self.ai_keywords = {
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'automation', 'chatgpt', 'gpt', 'algorithm', 'data science',
            'analytics', 'predictive', 'neural network', 'deep learning',
            'nlp', 'computer vision', 'proptech', 'digital transformation'
        }
    
    def detect_ai_mentions(self, text: str) -> Dict[str, any]:
        """
        Detect AI mentions in text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with ai_mentioned (bool) and mention_count (int)
            
        Example:
            >>> ai_analyzer = AIAdoptionAnalyzer()
            >>> result = ai_analyzer.detect_ai_mentions("using AI and machine learning")
            >>> print(result)
            {'ai_mentioned': True, 'mention_count': 2}
        """
        if not text or pd.isna(text):
            return {'ai_mentioned': False, 'mention_count': 0}
        
        text_lower = text.lower()
        
        # Count mentions
        mention_count = 0
        for keyword in self.ai_keywords:
            mention_count += text_lower.count(keyword)
        
        return {
            'ai_mentioned': mention_count > 0,
            'mention_count': mention_count
        }
    
    def analyze_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Analyze AI mentions for all reviews.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with AI mention columns
            
        Example:
            >>> ai_analyzer = AIAdoptionAnalyzer()
            >>> df = ai_analyzer.analyze_dataframe(df)
        """
        print(f"\n🤖 Analyzing AI adoption mentions...")
        
        # Detect AI mentions
        ai_data = df[text_column].apply(self.detect_ai_mentions)
        
        df = df.copy()
        df['ai_mentioned'] = [d['ai_mentioned'] for d in ai_data]
        df['ai_mention_count'] = [d['mention_count'] for d in ai_data]
        
        # Statistics
        total_mentions = df['ai_mentioned'].sum()
        pct = (total_mentions / len(df)) * 100
        
        print(f"  ✓ Found {total_mentions:,} reviews with AI mentions ({pct:.1f}%)")
        print(f"  ✓ Total AI keyword occurrences: {df['ai_mention_count'].sum():,}")
        
        return df
    
    def aggregate_by_time(self,
                         df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate AI mentions to company-month level.
        
        Args:
            df: Review-level DataFrame
            
        Returns:
            Aggregated DataFrame
            
        Example:
            >>> ai_monthly = ai_analyzer.aggregate_by_time(df)
            >>> print(ai_monthly.head())
        """
        print(f"\n📅 Aggregating AI mentions by company-month...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate
        agg_dict = {
            'ai_mentioned': 'sum',
            'ai_mention_count': 'sum',
        }
        
        monthly = df.groupby(['ticker', 'year_month']).agg({
            **agg_dict,
            'ticker': 'count'  # Total reviews
        }).reset_index()
        
        monthly.rename(columns={'ticker': 'review_count'}, inplace=True)
        
        # Calculate percentage
        monthly['ai_mention_pct'] = (monthly['ai_mentioned'] / monthly['review_count']) * 100
        
        # Convert period to string
        monthly['year_month'] = monthly['year_month'].astype(str)
        
        print(f"✓ Aggregated to {len(monthly):,} company-month observations")
        
        return monthly


if __name__ == '__main__':
    # Example usage
    print("Keyword Extraction Example")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "great benefit work life balance flexible schedule",
        "ai machine learning automation technology platform",
        "management leadership communication culture team",
        "salary compensation equity bonus competitive pay"
    ]
    
    # Initialize extractor
    extractor = KeywordExtractor(max_features=50, ngram_range=(1, 2))
    extractor.fit(texts)
    
    # Extract keywords
    keywords = extractor.get_top_keywords(texts, top_k=10)
    print("\nTop Keywords:")
    for i, (kw, score) in enumerate(keywords, 1):
        print(f"  {i:2d}. {kw:20s} ({score:.3f})")
    
    # Category classification
    print("\n" + "=" * 60)
    print("Category Classification Example")
    print("=" * 60)
    
    classifier = CategoryClassifier()
    for text in texts:
        result = classifier.classify_text(text)
        categories_found = [cat for cat, found in result.items() if found]
        print(f"\nText: {text}")
        print(f"Categories: {', '.join(categories_found) if categories_found else 'None'}")
    
    # AI adoption analysis
    print("\n" + "=" * 60)
    print("AI Adoption Analysis Example")
    print("=" * 60)
    
    ai_analyzer = AIAdoptionAnalyzer()
    for text in texts:
        result = ai_analyzer.detect_ai_mentions(text)
        print(f"\nText: {text}")
        print(f"AI mentioned: {result['ai_mentioned']} (count: {result['mention_count']})")