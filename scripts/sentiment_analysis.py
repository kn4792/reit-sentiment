#!/usr/bin/env python3
"""
Sentiment Analysis Script for REIT Glassdoor Reviews

Implements FinBERT-based sentiment analysis with support for:
- GPU acceleration (CUDA)
- Batch processing for memory efficiency
- Multiple sentiment metrics (FinBERT, Loughran-McDonald, TextBlob)
- Aggregation to company-month level

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
Version: 1.0

Usage:
    python scripts/sentiment_analysis.py --input data/processed/cleaned_reviews.csv
    python scripts/sentiment_analysis.py --input data/processed/*.csv --use-gpu
    python scripts/sentiment_analysis.py --input data/processed/cleaned.csv --batch-size 16
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    Uses ProsusAI/finbert model fine-tuned on financial phrasebank.
    Outputs: positive, negative, neutral probabilities + compound score.
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
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with keys: positive, negative, neutral, compound
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


class LoughranMcDonaldSentiment:
    """
    Loughran-McDonald financial dictionary-based sentiment.
    
    Uses word lists for positive/negative financial terms.
    Simpler but faster than transformer models.
    """
    
    def __init__(self):
        """Initialize with Loughran-McDonald word lists."""
        # Simplified financial positive/negative words
        # In production, load from actual LM dictionary files
        
        self.positive_words = {
            'profit', 'gain', 'growth', 'increase', 'strong', 'excellent',
            'improve', 'success', 'benefit', 'advance', 'positive', 'opportunity',
            'achieve', 'innovative', 'leading', 'competitive', 'valuable'
        }
        
        self.negative_words = {
            'loss', 'decline', 'decrease', 'weak', 'poor', 'fail', 'risk',
            'concern', 'problem', 'difficult', 'negative', 'challenge',
            'uncertainty', 'volatile', 'litigation', 'adverse', 'downturn'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Calculate LM sentiment score.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with pos_count, neg_count, net_score
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
        """
        print(f"\n📊 Calculating Loughran-McDonald sentiment...")
        
        results = df[text_column].apply(self.analyze)
        
        df['lm_positive_count'] = [r['pos_count'] for r in results]
        df['lm_negative_count'] = [r['neg_count'] for r in results]
        df['lm_net_score'] = [r['net_score'] for r in results]
        
        print("✓ LM sentiment complete")
        
        return df


def aggregate_to_monthly(df: pd.DataFrame, 
                        output_path: Path) -> pd.DataFrame:
    """
    Aggregate sentiment scores to company-month level.
    
    Computes mean, std, count for each company-month.
    
    Args:
        df: Review-level DataFrame with sentiment scores
        output_path: Path to save aggregated data
        
    Returns:
        Aggregated DataFrame
    """
    print("\n📅 Aggregating to company-month level...")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create year-month column
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Aggregate by company and month
    agg_dict = {
        'sentiment_compound': ['mean', 'std', 'count'],
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean',
        'lm_net_score': 'mean',
    }
    
    monthly = df.groupby(['ticker', 'year_month']).agg(agg_dict).reset_index()
    
    # Flatten column names
    monthly.columns = ['_'.join(col).strip('_') for col in monthly.columns.values]
    monthly.rename(columns={'ticker': 'ticker', 'year_month': 'year_month'}, 
                   inplace=True)
    
    # Convert period back to string
    monthly['year_month'] = monthly['year_month'].astype(str)
    
    # Save
    monthly.to_csv(output_path, index=False)
    print(f"✓ Saved {len(monthly):,} company-month observations → {output_path}")
    
    return monthly


def calculate_sentiment_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate summary statistics for sentiment scores.
    
    Args:
        df: DataFrame with sentiment columns
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    if 'sentiment_compound' in df.columns:
        stats['mean_compound'] = df['sentiment_compound'].mean()
        stats['std_compound'] = df['sentiment_compound'].std()
        stats['median_compound'] = df['sentiment_compound'].median()
    
    if 'sentiment_label' in df.columns:
        stats['pct_positive'] = (df['sentiment_label'] == 'positive').sum() / len(df) * 100
        stats['pct_negative'] = (df['sentiment_label'] == 'negative').sum() / len(df) * 100
        stats['pct_neutral'] = (df['sentiment_label'] == 'neutral').sum() / len(df) * 100
    
    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Sentiment analysis for REIT Glassdoor reviews using FinBERT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/03_sentiment_analysis.py --input data/processed/cleaned_reviews.csv
  
  # Use GPU acceleration
  python scripts/03_sentiment_analysis.py --input data/processed/cleaned.csv --use-gpu
  
  # Custom batch size
  python scripts/03_sentiment_analysis.py --input data/processed/cleaned.csv --batch-size 16
  
  # Skip monthly aggregation
  python scripts/03_sentiment_analysis.py --input data/processed/cleaned.csv --no-aggregate
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file with cleaned reviews'
    )
    parser.add_argument(
        '--output',
        default='data/results',
        help='Output directory (default: data/results)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration (CUDA)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--text-column',
        default='pros_cleaned',
        help='Column containing text to analyze (default: pros_cleaned)'
    )
    parser.add_argument(
        '--no-aggregate',
        action='store_true',
        help='Skip monthly aggregation'
    )
    parser.add_argument(
        '--skip-lm',
        action='store_true',
        help='Skip Loughran-McDonald dictionary analysis'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print("📂 LOADING DATA")
    print('='*60)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ File not found: {args.input}")
        return
    
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df):,} reviews from {input_path.name}")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Check for required columns
    if args.text_column not in df.columns:
        print(f"✗ Column '{args.text_column}' not found in data")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Initialize FinBERT analyzer
    print(f"\n{'='*60}")
    print("🤖 INITIALIZING FINBERT")
    print('='*60)
    
    try:
        finbert = FinBERTSentimentAnalyzer(
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"✗ Failed to initialize FinBERT: {e}")
        return
    
    # Analyze sentiment with FinBERT
    print(f"\n{'='*60}")
    print("🔍 FINBERT SENTIMENT ANALYSIS")
    print('='*60)
    
    df = finbert.analyze_dataframe(df, text_column=args.text_column)
    
    # Analyze with Loughran-McDonald
    if not args.skip_lm:
        print(f"\n{'='*60}")
        print("📊 LOUGHRAN-MCDONALD ANALYSIS")
        print('='*60)
        
        lm_analyzer = LoughranMcDonaldSentiment()
        df = lm_analyzer.analyze_dataframe(df, text_column=args.text_column)
    
    # Save review-level results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'sentiment_scores_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved review-level sentiment → {output_file}")
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("📊 SENTIMENT STATISTICS")
    print('='*60)
    
    stats = calculate_sentiment_stats(df)
    
    for key, value in stats.items():
        if 'pct' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Aggregate to monthly
    if not args.no_aggregate and 'ticker' in df.columns and 'date' in df.columns:
        print(f"\n{'='*60}")
        print("📅 MONTHLY AGGREGATION")
        print('='*60)
        
        monthly_file = output_dir / f'monthly_sentiment_{timestamp}.csv'
        monthly_df = aggregate_to_monthly(df, monthly_file)
        
        print(f"  • Companies: {monthly_df['ticker'].nunique()}")
        print(f"  • Time range: {monthly_df['year_month'].min()} to {monthly_df['year_month'].max()}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ SENTIMENT ANALYSIS COMPLETE")
    print('='*60)
    print(f"📝 Total reviews analyzed: {len(df):,}")
    print(f"💾 Output files saved to: {output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()