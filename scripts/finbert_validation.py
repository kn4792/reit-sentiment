#!/usr/bin/env python3
"""
Consolidated FinBERT Analysis & Model Evaluation Pipeline

This script performs end-to-end sentiment analysis and model evaluation:
1. Loads train/val/test datasets
2. Computes FinBERT sentiment scores for all splits
3. Trains classification and regression models
4. Evaluates on validation and test sets
5. Saves comprehensive results (metrics, predictions, summary tables)

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    python consolidated_sentiment_pipeline.py \
        --train data/processed/train.csv \
        --val data/processed/val.csv \
        --test data/processed/test.csv \
        --output data/results/consolidated_results \
        --text-column pros_cleaned
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
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
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        print(f"Loading FinBERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully\n")
        except Exception as e:
            print(f"Error loading model: {e}")
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
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        probs = predictions[0].cpu().numpy()
        
        return {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2]),
            'compound': float(probs[0] - probs[1])
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
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            valid_indices = [j for j, t in enumerate(batch) if t and not pd.isna(t)]
            valid_texts = [batch[j] for j in valid_indices]
            
            if not valid_texts:
                results.extend([{
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'compound': 0.0
                } for _ in batch])
                continue
            
            inputs = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions.cpu().numpy()
            
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
        print(f"Analyzing sentiment for {len(df):,} reviews...")
        print(f"Text column: {text_column}\n")
        
        texts = df[text_column].fillna('').tolist()
        
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size), 
                     desc="Processing batches"):
            batch = texts[i:i + self.batch_size]
            batch_results = self.predict_batch(batch)
            results.extend(batch_results)
        
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        
        labels = []
        for r in results:
            if r['positive'] > r['negative'] and r['positive'] > r['neutral']:
                labels.append('positive')
            elif r['negative'] > r['positive'] and r['negative'] > r['neutral']:
                labels.append('negative')
            else:
                labels.append('neutral')
        df['sentiment_label'] = labels
        
        print("Sentiment analysis complete\n")
        
        return df


class SentimentClassifier:
    """
    Sentiment-based classification model.
    Predicts rating category (high/medium/low) from sentiment features.
    """
    
    def __init__(self, model_type='logistic'):
        """
        Initialize classifier.
        
        Args:
            model_type: 'logistic' or 'random_forest'
        """
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df):
        """Extract features for classification."""
        features = []
        
        if 'sentiment_compound' in df.columns:
            features.append('sentiment_compound')
        if 'sentiment_positive' in df.columns:
            features.append('sentiment_positive')
        if 'sentiment_negative' in df.columns:
            features.append('sentiment_negative')
        
        X = df[features].fillna(0)
        
        return X, features
    
    def prepare_target(self, df):
        """Create target variable from rating."""
        if 'rating' not in df.columns:
            raise ValueError("Rating column not found")
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Create categories: low (1-3), medium (3-4), high (4-5)
        y = pd.cut(df['rating'], bins=[0, 3, 4, 5], labels=['low', 'medium', 'high'])
        
        return y
    
    def train(self, train_df):
        """Train the model."""
        X_train, self.feature_names = self.prepare_features(train_df)
        y_train = self.prepare_target(train_df)
        
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"Training {self.model_type} classifier...")
        print(f"Features: {self.feature_names}")
        print(f"Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        
        print(f"Training accuracy: {train_acc:.4f}\n")
        
        return train_acc
    
    def evaluate(self, eval_df, split_name='Test'):
        """Evaluate the model."""
        X_eval, _ = self.prepare_features(eval_df)
        y_true = self.prepare_target(eval_df)
        
        valid_idx = ~y_true.isna()
        X_eval = X_eval[valid_idx]
        y_true = y_true[valid_idx]
        
        y_pred = self.model.predict(X_eval)
        
        metrics = {
            'split': split_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'n_samples': len(y_true)
        }
        
        return metrics, y_true, y_pred


class PerformancePredictor:
    """
    Regression model to predict REIT performance from sentiment.
    """
    
    def __init__(self, model_type='linear'):
        """
        Initialize regressor.
        
        Args:
            model_type: 'linear' or 'random_forest'
        """
        self.model_type = model_type
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(random_state=42, n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df):
        """Extract features for regression."""
        features = []
        
        if 'sentiment_compound' in df.columns:
            features.append('sentiment_compound')
        if 'sentiment_positive' in df.columns:
            features.append('sentiment_positive')
        if 'sentiment_negative' in df.columns:
            features.append('sentiment_negative')
        
        X = df[features].fillna(0)
        
        return X, features
    
    def train(self, train_df, target_col='rating'):
        """Train the model."""
        X_train, self.feature_names = self.prepare_features(train_df)
        
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y_train = pd.to_numeric(train_df[target_col], errors='coerce')
        
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"Training {self.model_type} regressor...")
        print(f"Features: {self.feature_names}")
        print(f"Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        
        print(f"Training R-squared: {train_r2:.4f}\n")
        
        return train_r2
    
    def evaluate(self, eval_df, target_col='rating', split_name='Test'):
        """Evaluate the model."""
        X_eval, _ = self.prepare_features(eval_df)
        y_true = pd.to_numeric(eval_df[target_col], errors='coerce')
        
        valid_idx = ~y_true.isna()
        X_eval = X_eval[valid_idx]
        y_true = y_true[valid_idx]
        
        y_pred = self.model.predict(X_eval)
        
        metrics = {
            'split': split_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        return metrics, y_true, y_pred


def run_consolidated_pipeline(train_file: str, 
                              val_file: str, 
                              test_file: str,
                              output_dir: str,
                              text_column: str = 'pros_cleaned',
                              use_gpu: bool = False,
                              batch_size: int = 8):
    """
    Run complete sentiment analysis and model evaluation pipeline.
    
    Args:
        train_file: Path to training data CSV
        val_file: Path to validation data CSV
        test_file: Path to test data CSV
        output_dir: Directory to save results
        text_column: Column containing text to analyze
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for FinBERT processing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("CONSOLIDATED SENTIMENT ANALYSIS & MODEL EVALUATION PIPELINE")
    print("="*80)
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"Test file: {test_file}")
    print(f"Output directory: {output_dir}")
    print(f"Text column: {text_column}")
    print("="*80 + "\n")
    
    # Load data
    print("STEP 1: LOADING DATA")
    print("-"*80)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Train: {len(train_df):,} samples")
    print(f"Val:   {len(val_df):,} samples")
    print(f"Test:  {len(test_df):,} samples\n")
    
    # Initialize FinBERT analyzer
    print("STEP 2: INITIALIZING FINBERT")
    print("-"*80)
    analyzer = FinBERTSentimentAnalyzer(
        use_gpu=use_gpu,
        batch_size=batch_size
    )
    
    # Analyze sentiment for all splits
    print("STEP 3: COMPUTING SENTIMENT SCORES")
    print("-"*80)
    
    print("Processing training set...")
    train_df = analyzer.analyze_dataframe(train_df, text_column=text_column)
    
    print("Processing validation set...")
    val_df = analyzer.analyze_dataframe(val_df, text_column=text_column)
    
    print("Processing test set...")
    test_df = analyzer.analyze_dataframe(test_df, text_column=text_column)
    
    # Save sentiment-scored datasets
    train_df.to_csv(output_dir / f'train_with_sentiment_{timestamp}.csv', index=False)
    val_df.to_csv(output_dir / f'val_with_sentiment_{timestamp}.csv', index=False)
    test_df.to_csv(output_dir / f'test_with_sentiment_{timestamp}.csv', index=False)
    print(f"Saved sentiment-scored datasets to {output_dir}\n")
    
    # Train and evaluate classification model
    print("STEP 4: TRAINING CLASSIFICATION MODEL")
    print("-"*80)
    
    classifier = SentimentClassifier(model_type='logistic')
    train_acc = classifier.train(train_df)
    
    val_metrics_clf, _, _ = classifier.evaluate(val_df, 'Validation')
    print(f"Validation accuracy: {val_metrics_clf['accuracy']:.4f}")
    print(f"Validation F1 (macro): {val_metrics_clf['f1_macro']:.4f}\n")
    
    test_metrics_clf, y_true_clf, y_pred_clf = classifier.evaluate(test_df, 'Test')
    print(f"Test accuracy: {test_metrics_clf['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_metrics_clf['f1_macro']:.4f}\n")
    
    # Train and evaluate regression model
    print("STEP 5: TRAINING REGRESSION MODEL")
    print("-"*80)
    
    regressor = PerformancePredictor(model_type='linear')
    train_r2 = regressor.train(train_df, target_col='rating')
    
    val_metrics_reg, _, _ = regressor.evaluate(val_df, target_col='rating', split_name='Validation')
    print(f"Validation R-squared: {val_metrics_reg['r2']:.4f}")
    print(f"Validation RMSE: {val_metrics_reg['rmse']:.4f}\n")
    
    test_metrics_reg, y_true_reg, y_pred_reg = regressor.evaluate(test_df, target_col='rating', split_name='Test')
    print(f"Test R-squared: {test_metrics_reg['r2']:.4f}")
    print(f"Test RMSE: {test_metrics_reg['rmse']:.4f}\n")
    
    # Compile results
    print("STEP 6: SAVING RESULTS")
    print("-"*80)
    
    results = {
        'classification': {
            'train': {'accuracy': train_acc},
            'val': val_metrics_clf,
            'test': test_metrics_clf
        },
        'regression': {
            'train': {'r2': train_r2},
            'val': val_metrics_reg,
            'test': test_metrics_reg
        }
    }
    
    # Save JSON results
    results_file = output_dir / f'model_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {results_file}")
    
    # Save CSV summary
    summary_data = []
    for exp_name, exp_results in results.items():
        for split, metrics in exp_results.items():
            row = {'experiment': exp_name, 'split': split}
            row.update(metrics)
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f'model_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary table to {summary_file}")
    
    # Save detailed classification report
    clf_report = classification_report(y_true_clf, y_pred_clf, output_dict=True)
    clf_report_df = pd.DataFrame(clf_report).transpose()
    clf_report_file = output_dir / f'classification_report_{timestamp}.csv'
    clf_report_df.to_csv(clf_report_file)
    print(f"Saved classification report to {clf_report_file}")
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(y_true_clf, y_pred_clf)
    conf_df = pd.DataFrame(conf_matrix, 
                          index=['True_low', 'True_medium', 'True_high'],
                          columns=['Pred_low', 'Pred_medium', 'Pred_high'])
    conf_file = output_dir / f'confusion_matrix_{timestamp}.csv'
    conf_df.to_csv(conf_file)
    print(f"Saved confusion matrix to {conf_file}")
    
    # Print final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print("\nClassification Results:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_metrics_clf['accuracy']:.4f}")
    print(f"  Test Accuracy:  {test_metrics_clf['accuracy']:.4f}")
    print(f"  Test F1:        {test_metrics_clf['f1_macro']:.4f}")
    
    print("\nRegression Results:")
    print(f"  Train R-squared: {train_r2:.4f}")
    print(f"  Val R-squared:   {val_metrics_reg['r2']:.4f}")
    print(f"  Test R-squared:  {test_metrics_reg['r2']:.4f}")
    print(f"  Test RMSE:       {test_metrics_reg['rmse']:.4f}")
    
    print("\nAll outputs saved to:", output_dir)
    print("="*80)
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Consolidated sentiment analysis and model evaluation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python consolidated_sentiment_pipeline.py \\
      --train data/processed/train.csv \\
      --val data/processed/val.csv \\
      --test data/processed/test.csv \\
      --output data/results/consolidated_results
  
  # With GPU acceleration
  python consolidated_sentiment_pipeline.py \\
      --train data/processed/train.csv \\
      --val data/processed/val.csv \\
      --test data/processed/test.csv \\
      --output data/results/consolidated_results \\
      --use-gpu \\
      --batch-size 16
  
  # Custom text column
  python consolidated_sentiment_pipeline.py \\
      --train data/processed/train.csv \\
      --val data/processed/val.csv \\
      --test data/processed/test.csv \\
      --output data/results/consolidated_results \\
      --text-column cons_cleaned
        """
    )
    
    parser.add_argument(
        '--train',
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--val',
        required=True,
        help='Path to validation data CSV'
    )
    parser.add_argument(
        '--test',
        required=True,
        help='Path to test data CSV'
    )
    parser.add_argument(
        '--output',
        default='data/results/consolidated_results',
        help='Output directory (default: data/results/consolidated_results)'
    )
    parser.add_argument(
        '--text-column',
        default='pros_cleaned',
        help='Column containing text to analyze (default: pros_cleaned)'
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
    
    args = parser.parse_args()
    
    # Validate input files exist
    for file_path in [args.train, args.val, args.test]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return 1
    
    try:
        run_consolidated_pipeline(
            train_file=args.train,
            val_file=args.val,
            test_file=args.test,
            output_dir=args.output,
            text_column=args.text_column,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )
        return 0
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())