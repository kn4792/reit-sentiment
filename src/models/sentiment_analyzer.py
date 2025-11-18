"""
Sentiment Analysis Module for REIT Reviews

Implements FinBERT-based sentiment analysis for financial text.

Author: Konain Niaz (kn4792@rit.edu)

"""

import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses ProsusAI/finbert model fine-tuned on financial phrasebank.
    Outputs: positive, negative, neutral probabilities + compound score.
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Computing device (cuda/cpu)
        batch_size: Number of texts to process at once
        tokenizer: FinBERT tokenizer
        model: FinBERT model
    
    Example:
        >>> analyzer = SentimentAnalyzer(use_gpu=True)
        >>> sentiment = analyzer.predict("Great company with excellent benefits")
        >>> print(sentiment['compound'])
        0.75
    """
    
    def __init__(self,
                 model_name: str = 'ProsusAI/finbert',
                 use_gpu: bool = False,
                 batch_size: int = 8):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            use_gpu: Whether to use CUDA GPU acceleration
            batch_size: Number of texts to process at once
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Set device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        print(f"🤖 Loading FinBERT model: {model_name}")
        print(f"💻 Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with keys: positive, negative, neutral, compound
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.predict("Great benefits and culture")
            >>> print(result)
            {'positive': 0.85, 'negative': 0.05, 'neutral': 0.10, 'compound': 0.80}
        """
        if not text or pd.isna(text):
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to probabilities
        probs = predictions[0].cpu().numpy()
        
        # FinBERT outputs: [positive, negative, neutral]
        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'compound': float(probs[0] - probs[1])  # Compound score
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> texts = ["Great company", "Bad management"]
            >>> results = analyzer.predict_batch(texts)
            >>> print(results[0]['compound'])
            0.75
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Filter empty texts
            valid_indices = [j for j, t in enumerate(batch) if t and not pd.isna(t)]
            valid_texts = [batch[j] for j in valid_indices]
            
            if not valid_texts:
                # All empty - return neutral
                results.extend([{
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'compound': 0.0
                } for _ in batch])
                continue
            
            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to probabilities
            probs = predictions.cpu().numpy()
            
            # Build results for this batch
            batch_results = []
            valid_idx = 0
            
            for j in range(len(batch)):
                if j in valid_indices:
                    p = probs[valid_idx]
                    batch_results.append({
                        'positive': float(p[0]),
                        'negative': float(p[1]),
                        'neutral': float(p[2]),
                        'compound': float(p[0] - p[1])
                    })
                    valid_idx += 1
                else:
                    batch_results.append({
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 1.0,
                        'compound': 0.0
                    })
            
            results.extend(batch_results)
        
        return results
    
    def analyze_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Analyze sentiment for all rows in DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment columns added
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df = analyzer.analyze_dataframe(df, text_column='pros_cleaned')
            >>> print(df[['sentiment_compound', 'sentiment_label']].head())
        """
        print(f"\n🔍 Analyzing sentiment for {len(df):,} reviews...")
        print(f"📝 Text column: {text_column}")
        
        # Extract texts
        texts = df[text_column].fillna('').tolist()
        
        # Predict with progress bar
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                     desc="Processing batches"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.predict_batch(batch)
            results.extend(batch_results)
        
        # Add sentiment columns
        df = df.copy()
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        
        # Add discrete label (highest probability)
        labels = []
        for r in results:
            if r['positive'] > r['negative'] and r['positive'] > r['neutral']:
                labels.append('positive')
            elif r['negative'] > r['positive'] and r['negative'] > r['neutral']:
                labels.append('negative')
            else:
                labels.append('neutral')
        df['sentiment_label'] = labels
        
        print("✓ Sentiment analysis complete")
        
        return df
    
    def get_sentiment_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for sentiment scores.
        
        Args:
            df: DataFrame with sentiment columns
            
        Returns:
            Dictionary of statistics
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> stats = analyzer.get_sentiment_stats(df)
            >>> print(stats['mean_compound'])
            0.35
        """
        stats = {}
        
        if 'sentiment_compound' in df.columns:
            stats['mean_compound'] = float(df['sentiment_compound'].mean())
            stats['std_compound'] = float(df['sentiment_compound'].std())
            stats['median_compound'] = float(df['sentiment_compound'].median())
            stats['min_compound'] = float(df['sentiment_compound'].min())
            stats['max_compound'] = float(df['sentiment_compound'].max())
        
        if 'sentiment_label' in df.columns:
            total = len(df)
            stats['pct_positive'] = (df['sentiment_label'] == 'positive').sum() / total * 100
            stats['pct_negative'] = (df['sentiment_label'] == 'negative').sum() / total * 100
            stats['pct_neutral'] = (df['sentiment_label'] == 'neutral').sum() / total * 100
        
        return stats


class LoughranMcDonaldSentiment:
    """
    Loughran-McDonald financial dictionary-based sentiment.
    
    Uses word lists for positive/negative financial terms.
    Simpler but faster than transformer models.
    
    Attributes:
        positive_words: Set of positive financial terms
        negative_words: Set of negative financial terms
    
    Example:
        >>> lm = LoughranMcDonaldSentiment()
        >>> result = lm.analyze("Strong profit growth and excellent performance")
        >>> print(result['net_score'])
        0.25
    """
    
    def __init__(self):
        """Initialize with Loughran-McDonald word lists."""
        # Simplified financial positive/negative words
        # In production, load from actual LM dictionary files
        
        self.positive_words = {
            'profit', 'gain', 'growth', 'increase', 'strong', 'excellent',
            'improve', 'success', 'benefit', 'advance', 'positive', 'opportunity',
            'achieve', 'innovative', 'leading', 'competitive', 'valuable',
            'great', 'good', 'outstanding', 'superior', 'productive'
        }
        
        self.negative_words = {
            'loss', 'decline', 'decrease', 'weak', 'poor', 'fail', 'risk',
            'concern', 'problem', 'difficult', 'negative', 'challenge',
            'uncertainty', 'volatile', 'litigation', 'adverse', 'downturn',
            'bad', 'terrible', 'awful', 'worst', 'disappointing'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Calculate LM sentiment score.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with pos_count, neg_count, net_score
            
        Example:
            >>> lm = LoughranMcDonaldSentiment()
            >>> result = lm.analyze("profit growth and strong performance")
            >>> print(result)
            {'pos_count': 3, 'neg_count': 0, 'net_score': 0.5}
        """
        if not text or pd.isna(text):
            return {'pos_count': 0, 'neg_count': 0, 'net_score': 0.0}
        
        words = text.lower().split()
        
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        
        # Net score normalized by length
        total_words = len(words) if len(words) > 0 else 1
        net_score = (pos_count - neg_count) / total_words
        
        return {
            'pos_count': pos_count,
            'neg_count': neg_count,
            'net_score': net_score
        }
    
    def analyze_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Analyze sentiment for DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with LM sentiment columns
            
        Example:
            >>> lm = LoughranMcDonaldSentiment()
            >>> df = lm.analyze_dataframe(df)
            >>> print(df['lm_net_score'].mean())
            0.15
        """
        print(f"\n📊 Calculating Loughran-McDonald sentiment...")
        
        results = df[text_column].apply(self.analyze)
        
        df = df.copy()
        df['lm_positive_count'] = [r['pos_count'] for r in results]
        df['lm_negative_count'] = [r['neg_count'] for r in results]
        df['lm_net_score'] = [r['net_score'] for r in results]
        
        print("✓ LM sentiment complete")
        
        return df


if __name__ == '__main__':
    # Example usage
    print("Sentiment Analysis Example")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "Great company with excellent benefits and strong culture",
        "Bad management, poor communication, and declining morale",
        "Standard workplace with average conditions"
    ]
    
    # Initialize analyzer (using CPU for demo)
    print("\nInitializing FinBERT analyzer...")
    analyzer = SentimentAnalyzer(use_gpu=False, batch_size=2)
    
    # Analyze texts
    print("\nAnalyzing texts:")
    print("-" * 60)
    for text in texts:
        result = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['compound']:.3f} (P:{result['positive']:.3f}, N:{result['negative']:.3f})")
    
    # Loughran-McDonald example
    print("\n" + "=" * 60)
    print("Loughran-McDonald Sentiment Example")
    print("=" * 60)
    
    lm = LoughranMcDonaldSentiment()
    for text in texts:
        result = lm.analyze(text)
        print(f"\nText: {text}")
        print(f"LM Score: {result['net_score']:.3f} (Pos:{result['pos_count']}, Neg:{result['neg_count']})")