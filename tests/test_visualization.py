"""
Integration tests for Results Visualization (Model Metrics only)

This module tests visualization from model_summary_*.csv focusing on:
- Metrics barplot by split
- Metrics correlation heatmap
- Simple performance table generation
- Table and plot file formats

Author: Konain Niaz
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

@pytest.mark.integration
class TestModelMetricsVisualization:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Create minimal model summary CSV & output folders for testing."""
        self.plots_dir = tmp_path / "plots"
        self.tables_dir = tmp_path / "tables"
        self.plots_dir.mkdir()
        self.tables_dir.mkdir()
        # Dummy small summary matching model_summary_*.csv
        self.df = pd.DataFrame({
            'experiment': ['classification']*3 + ['regression']*3,
            'split': ['train', 'Validation', 'Test']*2,
            'accuracy': [0.81, 0.87, 0.93, np.nan, np.nan, np.nan],
            'precision_macro': [np.nan, 0.29, 0.47, np.nan, np.nan, np.nan],
            'recall_macro': [np.nan, 0.33, 0.5, np.nan, np.nan, np.nan],
            'f1_macro': [np.nan, 0.31, 0.48, np.nan, np.nan, np.nan],
            'n_samples': [np.nan, 15, 15, np.nan, 15, 15],
            'r2': [np.nan, np.nan, np.nan, 0.06, -0.12, -0.80],
            'mse': [np.nan, np.nan, np.nan, np.nan, 0.67, 0.11],
            'rmse': [np.nan, np.nan, np.nan, np.nan, 0.82, 0.33],
            'mae': [np.nan, np.nan, np.nan, np.nan, 0.51, 0.29],
        })
        self.results_file = tmp_path / "model_summary_test.csv"
        self.df.to_csv(self.results_file, index=False)
        yield

    def test_model_metrics_barplot(self):
        """Test barplot for model metrics by split"""
        df = pd.read_csv(self.results_file)
        metrics = ['accuracy','precision_macro','recall_macro','f1_macro','r2','mse','rmse','mae']
        longform = df.melt(id_vars=[c for c in ['experiment', 'split'] if c in df.columns],
                           value_vars=[col for col in metrics if col in df.columns],
                           var_name="Metric", value_name="Value")
        longform = longform[(~longform['Value'].isna()) & (longform['split'].notna())]
        if longform.empty:
            pytest.skip("No metrics available for barplot.")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=longform, x="Metric", y="Value", hue="split")
        plt.title("Model Metrics by Split")
        plt.tight_layout()
        plot_file = self.plots_dir / "model_metrics_bar.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists() and plot_file.stat().st_size > 0

    def test_correlation_heatmap(self):
        """Test correlation heatmap from numeric metrics"""
        df = pd.read_csv(self.results_file)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr()
        plt.figure(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plot_file = self.plots_dir / "metrics_corr_heatmap.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        assert plot_file.exists() and plot_file.stat().st_size > 0

    def test_csv_table_output(self):
        """Test simple metrics summary CSV output"""
        df = pd.read_csv(self.results_file)
        # Summarize some metrics by split
        summary = df.groupby('split').agg({
            'accuracy': 'mean',
            'f1_macro': 'mean',
            'r2': 'mean',
            'mse': 'mean'
        })
        table_file = self.tables_dir / "metrics_summary.csv"
        summary.to_csv(table_file)
        assert table_file.exists()
        loaded = pd.read_csv(table_file)
        assert not loaded.empty

    def test_latex_table_output(self):
        """Test LaTeX table export from summary"""
        df = pd.read_csv(self.results_file)
        summary = df.groupby('split').agg({
            'accuracy': 'mean',
            'f1_macro': 'mean',
            'r2': 'mean',
            'mse': 'mean'
        })
        latex_str = summary.to_latex(index=True)
        latex_file = self.tables_dir / "metrics_summary.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_str)
        assert latex_file.exists() and latex_file.stat().st_size > 0
        with open(latex_file, 'r') as f:
            content = f.read()
            assert '\\begin{tabular}' in content and '\\end{tabular}' in content

    def test_plot_formats(self):
        """Test plot file format export"""
        plt.figure()
        plt.plot([1, 2, 3], [2, 4, 6])
        for fmt in ['png', 'pdf', 'svg']:
            plot_file = self.plots_dir / f"fig1.{fmt}"
            plt.savefig(plot_file)
            assert plot_file.exists() and plot_file.stat().st_size > 0
        plt.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
