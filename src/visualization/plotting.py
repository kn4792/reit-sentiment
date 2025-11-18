#!/usr/bin/env python3
"""
Plotting Utilities for Sentiment EDA

Provides core plotting functions for REIT sentiment exploratory data analysis.
- Distribution plots
- Time-series visualizations
- Correlation matrices
- Scatter plots

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-18
Version: 1.0

Usage:
    from src.visualization.plotting import (
        generate_distribution_plots,
        generate_timeseries_plots,
        generate_correlation_matrix,
        generate_scatter_plots
    )
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def generate_distribution_plots(df: pd.DataFrame, columns, output_dir: Path, prefix="dist"):
    """
    Plot and save histograms of the specified columns.

    Args:
        df: DataFrame with data to plot.
        columns: List of column names to plot.
        output_dir: Directory to save plots.
        prefix: Output filename prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        outpath = output_dir / f"{prefix}_{col}.png"
        plt.savefig(outpath, dpi=150)
        plt.close()


def generate_timeseries_plots(df: pd.DataFrame, x_col, y_cols, output_dir: Path, prefix="timeseries"):
    """
    Plot and save time-series plots for specified columns.

    Args:
        df: DataFrame with data (must have x_col, usually date or period).
        x_col: Name of the time/date column.
        y_cols: List of value columns to plot over time.
        output_dir: Directory to save plots.
        prefix: Output filename prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for y in y_cols:
        plt.figure(figsize=(8, 4))
        plt.plot(df[x_col], df[y], marker='o')
        plt.title(f"{y} Over Time")
        plt.xlabel(x_col)
        plt.ylabel(y)
        plt.tight_layout()
        outpath = output_dir / f"{prefix}_{y}_over_{x_col}.png"
        plt.savefig(outpath, dpi=150)
        plt.close()


def generate_correlation_matrix(df: pd.DataFrame, columns, output_dir: Path, prefix="correlation"):
    """
    Plot and save a correlation heatmap of selected columns.

    Args:
        df: DataFrame with data.
        columns: List of columns to include in correlation.
        output_dir: Directory to save plots.
        prefix: Output filename prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    corr = df[columns].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    outpath = output_dir / f"{prefix}_heatmap.png"
    plt.savefig(outpath, dpi=150)
    plt.close()


def generate_scatter_plots(df: pd.DataFrame, x_col, y_col, output_dir: Path, prefix="scatter"):
    """
    Plot and save a scatter plot of two columns.

    Args:
        df: DataFrame with data.
        x_col: Name of X axis column.
        y_col: Name of Y axis column.
        output_dir: Directory to save plot.
        prefix: Output filename prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f"Scatter Plot: {y_col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    outpath = output_dir / f"{prefix}_{y_col}_vs_{x_col}.png"
    plt.savefig(outpath, dpi=150)
    plt.close()
