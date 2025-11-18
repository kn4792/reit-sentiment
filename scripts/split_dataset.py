#!/usr/bin/env python3
"""
Dataset Splitting Script

Splits cleaned review data into train/validation/test sets for reproducible experiments.
Stratified by company (ticker) to ensure balanced representation.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    python scripts/split_dataset.py --input data/processed/cleaned_all_reviews.csv \
                                    --train 0.7 --val 0.15 --test 0.15 \
                                    --outdir data/processed
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(input_file: Path,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 output_dir: Path = Path('data/processed'),
                 random_state: int = 42,
                 stratify_by: str = 'ticker'):
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        input_file: Path to cleaned combined dataset
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        output_dir: Directory to save split datasets
        random_state: Random seed for reproducibility
        stratify_by: Column to stratify by (default: ticker for balanced companies)
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    print(f"\n{'='*60}")
    print("DATASET SPLITTING")
    print('='*60)
    print(f"Input: {input_file}")
    print(f"Split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    print(f"Random seed: {random_state}")
    print(f"Stratify by: {stratify_by}")
    
    # Load data
    print(f"\n📂 Loading data...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} reviews")
    
    if stratify_by in df.columns:
        print(f"  • Unique {stratify_by}s: {df[stratify_by].nunique()}")
        stratify_col = df[stratify_by]
    else:
        print(f"⚠️  Column '{stratify_by}' not found, splitting without stratification")
        stratify_col = None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    # Calculate relative ratio for val/(val+test)
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    
    if stratify_col is not None:
        temp_stratify = temp_df[stratify_by]
    else:
        temp_stratify = None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=random_state,
        stratify=temp_stratify
    )
    
    # Report statistics
    print(f"\n📊 Split Statistics:")
    print(f"  • Train: {len(train_df):,} reviews ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  • Val:   {len(val_df):,} reviews ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  • Test:  {len(test_df):,} reviews ({len(test_df)/len(df)*100:.1f}%)")
    
    if stratify_by in df.columns:
        print(f"\n  Company distribution:")
        print(f"    Train: {train_df[stratify_by].nunique()} unique companies")
        print(f"    Val:   {val_df[stratify_by].nunique()} unique companies")
        print(f"    Test:  {test_df[stratify_by].nunique()} unique companies")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / 'train.csv'
    val_file = output_dir / 'val.csv'
    test_file = output_dir / 'test.csv'
    
    print(f"\n💾 Saving splits...")
    train_df.to_csv(train_file, index=False)
    print(f"  ✓ Train → {train_file}")
    
    val_df.to_csv(val_file, index=False)
    print(f"  ✓ Val   → {val_file}")
    
    test_df.to_csv(test_file, index=False)
    print(f"  ✓ Test  → {test_file}")
    
    # Save split metadata
    metadata = {
        'input_file': str(input_file),
        'total_reviews': len(df),
        'train_reviews': len(train_df),
        'val_reviews': len(val_df),
        'test_reviews': len(test_df),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_state': random_state,
        'stratify_by': stratify_by
    }
    
    metadata_file = output_dir / 'split_metadata.json'
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata → {metadata_file}")
    
    print(f"\n{'='*60}")
    print("✅ DATASET SPLIT COMPLETE")
    print('='*60)
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description='Split cleaned dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 70/15/15 split
  python scripts/split_dataset.py --input data/processed/cleaned_all_reviews.csv
  
  # Custom split ratios
  python scripts/split_dataset.py --input data/processed/cleaned_all_reviews.csv \\
                                  --train 0.8 --val 0.1 --test 0.1
  
  # Different stratification
  python scripts/split_dataset.py --input data/processed/cleaned_all_reviews.csv \\
                                  --stratify company
  
  # Custom output directory
  python scripts/split_dataset.py --input data/processed/cleaned_all_reviews.csv \\
                                  --outdir data/processed/splits
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to cleaned combined dataset CSV'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Train set proportion (default: 0.7)'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Validation set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Test set proportion (default: 0.15)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='data/processed',
        help='Output directory for split datasets (default: data/processed)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--stratify',
        type=str,
        default='ticker',
        help='Column to stratify by (default: ticker)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_file = Path(args.input)
    output_dir = Path(args.outdir)
    
    # Validate input exists
    if not input_file.exists():
        print(f"✗ Error: Input file not found: {input_file}")
        return
    
    # Run splitting
    try:
        split_dataset(
            input_file=input_file,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            output_dir=output_dir,
            random_state=args.seed,
            stratify_by=args.stratify
        )
    except Exception as e:
        print(f"\n✗ Error during splitting: {e}")
        raise


if __name__ == '__main__':
    main()