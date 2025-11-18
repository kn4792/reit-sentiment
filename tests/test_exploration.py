"""
Integration tests for Phase 3: Data Exploration & Visualization

This module tests exploratory data analysis including:
- Distribution plots
- Time-series analysis
- Word clouds (optional)
- Statistical summaries

Author: Konain Niaz
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

@pytest.mark.integration
class TestExplorationPipeline:
    """Integration tests for complete exploration pipeline"""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment with sample data"""
        self.exploration_dir = tmp_path / "exploration"
        self.exploration_dir.mkdir()

        # Create sample data with trends
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='W')
        n_samples = len(dates)
        self.sample_data = pd.DataFrame({
            'date': dates,
            'rating': np.random.uniform(2.5, 4.5, n_samples) + np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.3,
            'sentiment_score': np.random.uniform(-0.5, 0.8, n_samples) + np.sin(np.linspace(0, 4*np.pi, n_samples)) * 0.2,
            'pros': (['Great culture', 'Good benefits', 'Excellent team', 'Nice office',
                'Strong leadership', 'Career growth', 'Work-life balance', 'Competitive pay'] * ((n_samples // 8) + 1))[:n_samples],
            'cons': (['Long hours', 'Low pay', 'Poor management', 'Limited growth',
                'High pressure', 'Old technology', 'Bureaucracy', 'Unclear expectations'] * ((n_samples // 8) + 1))[:n_samples],

            'company': ['TestREIT'] * n_samples,
            'ticker': ['TEST'] * n_samples,
            'ai_related': np.random.choice([True, False], n_samples, p=[0.2, 0.8])
        })
        self.sample_data = self.sample_data.iloc[:n_samples]
        self.data_file = self.exploration_dir / "sample_data.csv"
        self.sample_data.to_csv(self.data_file, index=False)
        yield

    def test_data_loaded_for_exploration(self):
        """Test that data is loaded correctly for exploration"""
        df = pd.read_csv(self.data_file)
        assert len(df) > 0
        assert 'sentiment_score' in df.columns
        assert 'rating' in df.columns

    def test_distribution_plot_generation(self):
        """Test generation of distribution plots"""
        df = pd.read_csv(self.data_file)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['rating'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Ratings')
        plot_file = self.exploration_dir / "rating_distribution.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_sentiment_distribution_plot(self):
        """Test sentiment score distribution plot"""
        df = pd.read_csv(self.data_file)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(x=0, color='red', linestyle='--', label='Neutral')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Sentiment Scores')
        ax.legend()
        plot_file = self.exploration_dir / "sentiment_distribution.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()

    def test_time_series_plot_generation(self):
        """Test generation of time-series plots"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['date'], df['sentiment_score'], marker='o', markersize=3, linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Trend Over Time')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plot_file = self.exploration_dir / "sentiment_timeseries.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()

    def test_rolling_average_plot(self):
        """Test rolling average visualization"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['sentiment_rolling'] = df['sentiment_score'].rolling(window=4, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['date'], df['sentiment_score'], alpha=0.3, label='Raw')
        ax.plot(df['date'], df['sentiment_rolling'], linewidth=2, label='4-week MA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment with Rolling Average')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plot_file = self.exploration_dir / "sentiment_rolling_avg.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()

    def test_boxplot_generation(self):
        """Test boxplot generation for comparing groups"""
        df = pd.read_csv(self.data_file)
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='sentiment_score', by='ai_related', ax=ax)
        ax.set_xlabel('AI Related')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Score by AI Mention')
        plt.suptitle('')
        plot_file = self.exploration_dir / "sentiment_by_ai_boxplot.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()

    def test_statistical_summary_generation(self):
        """Test generation of statistical summaries"""
        df = pd.read_csv(self.data_file)
        summary = df[['rating', 'sentiment_score']].describe()
        summary_file = self.exploration_dir / "summary_statistics.csv"
        summary.to_csv(summary_file)
        assert summary_file.exists()
        loaded_summary = pd.read_csv(summary_file, index_col=0)
        assert 'rating' in loaded_summary.columns
        assert 'sentiment_score' in loaded_summary.columns
        assert 'mean' in loaded_summary.index
        assert 'std' in loaded_summary.index

    def test_monthly_aggregation(self):
        """Test monthly aggregation of data"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        monthly = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'rating': ['mean', 'std'],
            'ai_related': 'sum'
        }).reset_index()
        monthly_file = self.exploration_dir / "monthly_aggregated.csv"
        monthly.to_csv(monthly_file, index=False)
        assert monthly_file.exists()
        assert len(monthly) > 0

    def test_ai_mention_trend_over_time(self):
        """Test visualization of AI mention trends"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        monthly_ai = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            'ai_related': ['sum', 'count']
        })
        monthly_ai['ai_rate'] = monthly_ai[('ai_related', 'sum')] / monthly_ai[('ai_related', 'count')]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(monthly_ai.index, monthly_ai['ai_rate'], marker='o', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('AI Mention Rate')
        ax.set_title('AI Mention Rate Over Time')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plot_file = self.exploration_dir / "ai_mention_trend.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists()

    def test_complete_exploration_pipeline(self):
        """Test complete exploration pipeline end-to-end"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        # Summary statistics
        summary = df[['rating', 'sentiment_score']].describe()
        summary_file = self.exploration_dir / "summary_stats.csv"
        summary.to_csv(summary_file)
        # Generate distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(df['rating'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_title('Rating Distribution')
        axes[0].set_xlabel('Rating')
        axes[1].hist(df['sentiment_score'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].axvline(x=0, color='red', linestyle='--')
        axes[1].set_title('Sentiment Distribution')
        axes[1].set_xlabel('Sentiment Score')
        plt.tight_layout()
        combined_plot = self.exploration_dir / "exploration_summary.png"
        plt.savefig(combined_plot, dpi=100, bbox_inches='tight')
        plt.close()
        # Monthly aggregation
        monthly = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'rating': 'mean',
            'ai_related': 'sum'
        }).reset_index()
        monthly_file = self.exploration_dir / "monthly_trends.csv"
        monthly.to_csv(monthly_file, index=False)
        # Verify all outputs
        assert summary_file.exists()
        assert combined_plot.exists()
        assert monthly_file.exists()

        print(f"\nExploration Summary:")
        print(f"Mean rating: {df['rating'].mean():.2f}")
        print(f"Mean sentiment: {df['sentiment_score'].mean():.2f}")
        print(f"AI mention rate: {df['ai_related'].mean():.1%}")
        print("✓ Complete exploration pipeline executed successfully")


@pytest.mark.unit
class TestPlotGeneration:
    """Unit tests for individual plot generation functions"""
    def test_histogram_creation(self, tmp_path):
        data = np.random.normal(0, 1, 100)
        fig, ax = plt.subplots()
        ax.hist(data, bins=20)
        plot_file = tmp_path / "test_histogram.png"
        plt.savefig(plot_file)
        plt.close()
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_line_plot_creation(self, tmp_path):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plot_file = tmp_path / "test_lineplot.png"
        plt.savefig(plot_file)
        plt.close()
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_boxplot_creation(self, tmp_path):
        data = [np.random.normal(0, 1, 50) for _ in range(3)]
        fig, ax = plt.subplots()
        ax.boxplot(data)
        plot_file = tmp_path / "test_boxplot.png"
        plt.savefig(plot_file)
        plt.close()
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

@pytest.mark.unit
class TestStatisticalSummaries:
    """Unit tests for statistical summary generation"""
    def test_descriptive_statistics(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        stats = data.describe()
        assert 'mean' in stats.index
        assert 'std' in stats.index
        assert 'min' in stats.index
        assert 'max' in stats.index
        assert '50%' in stats.index # median

    def test_aggregation_functions(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        agg = df.groupby('group')['value'].agg(['mean', 'std', 'count'])
        assert len(agg) == 2
        assert 'mean' in agg.columns
        assert 'std' in agg.columns
        assert 'count' in agg.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
