#!/usr/bin/env python3
"""
Model Training & Evaluation Script

Trains and evaluates sentiment-based models on train/val/test splits.
Includes classification models, regression for performance prediction, and DiD analysis.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    python scripts/train_model.py --train data/results/sentiments_train/ \\
                                  --val data/results/sentiments_val \\
                                  --test data/results/sentiments_test \\
                                  --output data/results/model_results
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import json


def resolve_input_file(path):
    """If path is a dir, get latest sentiment_scores_*.csv inside it. Otherwise, return as-is (if file)."""
    path = Path(path)
    if path.is_file():
        return str(path)
    elif path.is_dir():
        files = list(path.glob("sentiment_scores_*.csv"))
        if not files:
            raise FileNotFoundError(f"No sentiment_scores_*.csv in {path}")
        return str(max(files, key=os.path.getctime))
    else:
        raise FileNotFoundError(f"{path} not found")

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
        
        # Sentiment features
        if 'sentiment_compound' in df.columns:
            features.append('sentiment_compound')
        if 'sentiment_positive' in df.columns:
            features.append('sentiment_positive')
        if 'sentiment_negative' in df.columns:
            features.append('sentiment_negative')
        
        # LM features
        if 'lm_net_score' in df.columns:
            features.append('lm_net_score')
        
        # AI features
        if 'ai_mentioned' in df.columns:
            features.append('ai_mentioned')
        
        X = df[features].fillna(0)
        
        return X, features
    
    def prepare_target(self, df):
        """Create target variable from rating."""
        if 'rating' not in df.columns:
            raise ValueError("Rating column not found")
        
        # Convert to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Create categories: low (1-3), medium (3-4), high (4-5)
        y = pd.cut(df['rating'], bins=[0, 3, 4, 5], labels=['low', 'medium', 'high'])
        
        return y
    
    def train(self, train_df):
        """Train the model."""
        X_train, self.feature_names = self.prepare_features(train_df)
        y_train = self.prepare_target(train_df)
        
        # Remove NaN targets
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"  Training {self.model_type} classifier...")
        print(f"  Features: {self.feature_names}")
        print(f"  Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        # Training performance
        y_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred)
        
        return train_acc
    
    def evaluate(self, eval_df, split_name='Test'):
        """Evaluate the model."""
        X_eval, _ = self.prepare_features(eval_df)
        y_true = self.prepare_target(eval_df)
        
        # Remove NaN targets
        valid_idx = ~y_true.isna()
        X_eval = X_eval[valid_idx]
        y_true = y_true[valid_idx]
        
        y_pred = self.model.predict(X_eval)
        
        # Metrics
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
        
        # Sentiment features
        if 'sentiment_compound' in df.columns:
            features.append('sentiment_compound')
        if 'lm_net_score' in df.columns:
            features.append('lm_net_score')
        if 'ai_mentioned' in df.columns:
            features.append('ai_mentioned')
        
        # Time features
        if 'date' in df.columns:
            df['month'] = pd.to_datetime(df['date']).dt.month
            df['year'] = pd.to_datetime(df['date']).dt.year
            features.extend(['month', 'year'])
        
        X = df[features].fillna(0)
        
        return X, features
    
    def train(self, train_df, target_col='rating'):
        """Train the model."""
        X_train, self.feature_names = self.prepare_features(train_df)
        
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y_train = pd.to_numeric(train_df[target_col], errors='coerce')
        
        # Remove NaN
        valid_idx = ~y_train.isna()
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        
        print(f"  Training {self.model_type} regressor...")
        print(f"  Features: {self.feature_names}")
        print(f"  Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        # Training performance
        y_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        
        return train_r2
    
    def evaluate(self, eval_df, target_col='rating', split_name='Test'):
        """Evaluate the model."""
        X_eval, _ = self.prepare_features(eval_df)
        y_true = pd.to_numeric(eval_df[target_col], errors='coerce')
        
        # Remove NaN
        valid_idx = ~y_true.isna()
        X_eval = X_eval[valid_idx]
        y_true = y_true[valid_idx]
        
        y_pred = self.model.predict(X_eval)
        
        # Metrics
        metrics = {
            'split': split_name,
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'n_samples': len(y_true)
        }
        
        return metrics, y_true, y_pred


def run_experiments(train_file, val_file, test_file, output_dir):
    """
    Run all experiments.
    
    Args:
        train_file: Path to training data
        val_file: Path to validation data
        test_file: Path to test data
        output_dir: Directory to save results
    """
    print(f"\n{'='*60}")
    print("MODEL TRAINING & EVALUATION")
    print('='*60)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"  ✓ Train: {len(train_df):,} samples")
    print(f"  ✓ Val:   {len(val_df):,} samples")
    print(f"  ✓ Test:  {len(test_df):,} samples")
    
    results = {}
    
    # Experiment 1: Rating Classification
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: RATING CLASSIFICATION")
    print('='*60)
    
    classifier = SentimentClassifier(model_type='logistic')
    train_acc = classifier.train(train_df)
    print(f"  Training accuracy: {train_acc:.4f}")
    
    val_metrics, _, _ = classifier.evaluate(val_df, 'Validation')
    print(f"  Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
    
    test_metrics, y_true, y_pred = classifier.evaluate(test_df, 'Test')
    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    results['classification'] = {
        'train': {'accuracy': train_acc},
        'val': val_metrics,
        'test': test_metrics
    }
    
    # Experiment 2: Rating Regression
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: RATING PREDICTION (REGRESSION)")
    print('='*60)
    
    regressor = PerformancePredictor(model_type='linear')
    train_r2 = regressor.train(train_df, target_col='rating')
    print(f"  Training R²: {train_r2:.4f}")
    
    val_metrics, _, _ = regressor.evaluate(val_df, target_col='rating', split_name='Validation')
    print(f"  Validation R²: {val_metrics['r2']:.4f}")
    print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
    
    test_metrics, _, _ = regressor.evaluate(test_df, target_col='rating', split_name='Test')
    print(f"  Test R²: {test_metrics['r2']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    
    results['regression'] = {
        'train': {'r2': train_r2},
        'val': val_metrics,
        'test': test_metrics
    }
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    results_file = output_dir / f'model_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved → {results_file}")
    
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
    print(f"💾 Summary saved → {summary_file}")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print('='*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate sentiment-based models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--train',
        type=str,
        required=True,
        help='Path to training data directory or file with sentiment scores'
    )
    parser.add_argument(
        '--val',
        type=str,
        required=True,
        help='Path to validation data directory or file with sentiment scores'
    )
    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test data directory or file with sentiment scores'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results',
        help='Output directory for results (default: data/results)'
    )

    args = parser.parse_args()

    train_file = resolve_input_file(args.train)
    val_file = resolve_input_file(args.val)
    test_file = resolve_input_file(args.test)

    for file_path in [train_file, val_file, test_file]:
        if not Path(file_path).exists():
            print(f"✗ Error: File not found: {file_path}")
            return

    try:
        run_experiments(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            output_dir=args.output
        )
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise


if __name__ == '__main__':
    main()