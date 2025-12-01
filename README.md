# REIT Employee Sentiment Analysis Pipeline

**Author:** Konain Niaz (kn4792@rit.edu)  
**Advisors:** Dr. Debanjana Dey, Dr. Travis Desell  
**Institution:** Rochester Institute of Technology  

---

## 1. Project Overview

This project analyzes the relationship between employee sentiment and REIT (Real Estate Investment Trust) performance using Glassdoor reviews and FinBERT sentiment analysis. The pipeline examines whether ChatGPT's launch (November 30, 2022) moderated the sentiment-performance relationship, with particular focus on technology-intensive REITs (data centers, telecommunications).

The methodology applies FinBERT (Financial BERT) for sentiment scoring, aggregates reviews to firm-year level, and uses difference-in-differences and triple-difference specifications to test whether:
1. Employee sentiment predicts REIT performance
2. ChatGPT's launch amplified this relationship
3. The effect is stronger for technology-intensive REITs exposed to AI infrastructure demand

---

## 2. System Requirements

### Required Software
- **Python**: 3.8 or higher
- **Recommended**: virtualenv or conda for environment management

### Hardware Recommendations
- **RAM**: 8GB minimum (16GB recommended for faster processing)
- **Storage**: 5GB available space
- **GPU**: CUDA-compatible GPU recommended for FinBERT (optional, falls back to CPU)

### Python Dependencies
- pandas, numpy, scipy
- torch, transformers (for FinBERT)
- statsmodels (for panel regressions)
- matplotlib, seaborn (for visualizations)
- tqdm (for progress bars)

---

## 3. Installation

### Clone Repository & Set Up Environment
```bash
# Clone repository
git clone https://github.com/kn4792/reit-sentiment.git
cd reit-sentiment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Sample `requirements.txt`
```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0
torch>=2.0.0
transformers>=4.30.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 4. Directory Structure

```
reit-sentiment-analysis/
│
├── config/
│   └── reit_companies.json      # List of REITs with metadata
│
├── data/
│   ├── raw/
│   │   └── all_reit_reviews_merged.csv  # Input: 28,252 Glassdoor reviews
│   │
│   ├── processed/
│   │   ├── finbert_sentiment_scores.csv # Stage 1 output
│   │   └── firm_year_sentiment.csv      # Stage 2 output
│   │
│   └── results/
│       ├── analysis_ready_dataset.csv   # Final dataset for regressions
│       ├── variable_codebook.md         # Variable documentation
│       ├── sample_merge_and_regression.py
│       │
│       └── descriptive_stats/
│           ├── figures/                 # 7 publication-ready figures
│           │   ├── correlation_heatmap.png
│           │   ├── time_trends.png
│           │   ├── pre_post_chatgpt_comparison.png
│           │   ├── sentiment_by_property_type.png
│           │   ├── tech_vs_traditional_comparison.png
│           │   ├── sentiment_distributions.png
│           │   └── review_volume_trend.png
│           │
│           ├── summary_statistics.csv
│           ├── stats_by_property_type.csv
│           ├── stats_pre_post_chatgpt.csv
│           ├── stats_by_year.csv
│           ├── correlation_matrix.csv
│           └── analysis_summary_report.txt
│
├── scripts/
│   ├── finbert_sentiment.py      # FinBERT sentiment extraction
│   ├── aggregate_firm_year.py    # Firm-year aggregation
│   ├── descriptive_analysis.py   # Descriptive stats & visualization
│   └── export_analysis_ready.py  # Export final dataset
│
├── .gitignore                            # Git ignore file
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 5. Pipeline Execution

### Quick Start (Complete Pipeline)

```bash
# 1. Setup directories
mkdir -p data/raw data/processed data/results scripts

# 2. Place your data
cp all_reit_reviews_merged.csv data/raw/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run 4-stage pipeline (Total time: ~45 minutes)
cd scripts/
python finbert_sentiment.py      # 15-30 min: FinBERT processing
python aggregate_firm_year.py    # 2 min: Aggregation
python descriptive_analysis.py   # 3 min: Descriptive analysis
python export_analysis_ready.py  # 1 min: Final export

# 5. Result
# → data/results/analysis_ready_dataset.csv (ready to merge with performance data!)
```

---

### Phase 1: FinBERT Sentiment Extraction

```bash
python finbert_sentiment.py
```

**Input**: `data/raw/all_reit_reviews_merged.csv` (28,252 reviews)  
**Output**: `data/processed/finbert_sentiment_scores.csv`

**What it does:**
1. Loads all Glassdoor reviews (pros + cons)
2. Downloads and initializes FinBERT model (ProsusAI/finbert)
3. Processes reviews in batches (default: 32 reviews per batch)
4. Extracts sentiment probabilities and scores for each review
5. Creates sentiment score: positive_prob - negative_prob (range: -1 to 1)

**Processing details:**
- Combines pros and cons with semantic labels: "Positive aspects: [pros] Negative aspects: [cons]"
- Handles missing values (converts NaN to empty string)
- Filters out completely empty reviews
- Uses GPU if available, falls back to CPU
- Shows progress bar during processing

**Expected output:**
```
Sentiment Distribution:
  positive:  ~12,000-14,000 reviews (45-50%)
  negative:   ~7,000-9,000 reviews (25-30%)
  neutral:    ~6,000-8,000 reviews (20-25%)

Mean sentiment score: ~0.00 to 0.15
```

**Runtime**: 15-30 minutes (GPU), 45-90 minutes (CPU)

---

### Phase 2: Firm-Year Aggregation

```bash
python aggregate_firm_year.py
```

**Input**: `data/processed/finbert_sentiment_scores.csv`  
**Output**: `data/processed/firm_year_sentiment.csv`

**What it does:**
1. Groups reviews by ticker and year
2. Calculates aggregate sentiment metrics per firm-year
3. Creates POST_CHATGPT treatment variable (1 if date > Nov 30, 2022)
4. Adds property type classifications (tech-intensive, datacenter_REIT)
5. Filters to minimum 3 reviews per firm-year for reliability

**Aggregation includes:**
- Mean, median, std dev, min, max sentiment
- Sentiment dispersion (within-firm-year disagreement)
- Sentiment agreement (% agreeing with majority category)
- Mean Glassdoor rating
- Review count
- Property type indicators

**Expected output:**
```
Firm-year observations: ~900-1,000
Unique firms: 115
Year range: 2014-2025
Pre-ChatGPT firm-years: ~650-700 (70-75%)
Post-ChatGPT firm-years: ~250-300 (25-30%)
```

**Runtime**: 1-2 minutes

---

### Phase 3: Descriptive Analysis

```bash
python descriptive_analysis.py
```

**Input**: `data/processed/firm_year_sentiment.csv`  
**Output**: `data/results/descriptive_stats/` (tables + figures)

**What it does:**
1. **Summary Statistics**: Creates 4 summary tables (overall, by property type, pre/post ChatGPT, by year)
2. **Correlation Analysis**: Correlation matrix with heatmap visualization
3. **Time Trends**: Sentiment and rating trends over time
4. **Pre/Post ChatGPT**: Statistical comparisons with t-tests
5. **Property Type Analysis**: Tech-intensive vs traditional REITs
6. **Visualizations**: 7 publication-ready figures (300 DPI PNG)

**Generated files:**
- **Tables** (CSV format):
  - summary_statistics.csv
  - stats_by_property_type.csv
  - stats_pre_post_chatgpt.csv
  - stats_by_year.csv
  - correlation_matrix.csv

- **Figures** (PNG, 300 DPI):
  - correlation_heatmap.png
  - time_trends.png
  - review_volume_trend.png
  - pre_post_chatgpt_comparison.png
  - sentiment_by_property_type.png
  - tech_vs_traditional_comparison.png
  - sentiment_distributions.png

- **Report**:
  - analysis_summary_report.txt

**Runtime**: 2-3 minutes

---



## 6. Merging with Performance Data

After completing the 4-stage pipeline, merge with your REIT performance data:

```python
import pandas as pd

# Load datasets
sentiment = pd.read_csv('data/results/analysis_ready_dataset.csv')
performance = pd.read_csv('your_reit_performance_data.csv')  # Must have 'ticker' and 'year'

# Standardize tickers (important!)
sentiment['ticker'] = sentiment['ticker'].str.upper().str.strip()
performance['ticker'] = performance['ticker'].str.upper().str.strip()

# Ensure years are integers
sentiment['year'] = sentiment['year'].astype(int)
performance['year'] = performance['year'].astype(int)

# Merge on ticker and year
merged = performance.merge(sentiment, on=['ticker', 'year'], how='inner')

# Check merge quality
print(f"Performance data: {len(performance)} rows")
print(f"Sentiment data: {len(sentiment)} rows")
print(f"Merged data: {len(merged)} rows")
print(f"Match rate: {len(merged)/len(performance)*100:.1f}%")
```

**Expected merge rate**: 60-80% (depends on your performance data coverage)

---

## 7. Regression Analysis

### Model 1: Baseline - Does sentiment predict performance?

```python
import statsmodels.formula.api as smf

# Baseline model
model1 = smf.ols('returns ~ sentiment + rating + review_count + C(year) + C(ticker)',
                 data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

print(model1.summary())
```

**Research Question**: Does employee sentiment predict REIT performance?  
**Hypothesis**: β_sentiment > 0

---

### Model 2: Difference-in-Differences - Did ChatGPT moderate the relationship?

```python
# DiD model
model2 = smf.ols('''returns ~ sentiment * POST_CHATGPT + 
                    rating + review_count + 
                    C(year) + C(ticker)''',
                 data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

print(model2.summary())
```

**Research Question**: Did ChatGPT's launch change the sentiment-performance relationship?  
**Hypothesis**: β_interaction > 0 (sentiment matters more post-ChatGPT)

**Interpretation**:
- `sentiment`: Effect pre-ChatGPT
- `sentiment:POST_CHATGPT`: Additional effect post-ChatGPT
- Total post-ChatGPT effect: β_sentiment + β_interaction

---

### Model 3: Triple-Difference - Stronger effect for tech-intensive REITs?

```python
# Triple-difference model
model3 = smf.ols('''returns ~ sentiment * tech_intensive * POST_CHATGPT +
                    rating + review_count +
                    C(year) + C(ticker)''',
                 data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

print(model3.summary())
```

**Research Question**: Is the effect stronger for technology-intensive REITs?  
**Hypothesis**: β_three_way_interaction > 0

**Economic Story**: 
> "For data center REITs post-ChatGPT, employee sentiment became more predictive of performance because these firms' value depends on capturing AI infrastructure demand."

---

### Robustness Checks

```python
# 1. Alternative DV: FFO instead of returns
model_ffo = smf.ols('ffo ~ sentiment * POST_CHATGPT + controls + FE',
                    data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

# 2. Exclude outliers
merged_trim = merged[(merged['sentiment'] > -0.5) & (merged['sentiment'] < 0.5)]
model_robust = smf.ols('returns ~ sentiment * POST_CHATGPT + controls + FE',
                      data=merged_trim).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

# 3. Alternative sentiment measure
model_median = smf.ols('returns ~ sentiment_median * POST_CHATGPT + controls + FE',
                       data=merged).fit(cov_type='cluster', cov_kwds={'groups': merged['ticker']})

# 4. Placebo test: Pre-period only with fake treatment
pre_only = merged[merged['year'] < 2023]
pre_only['PLACEBO'] = (pre_only['year'] > 2020).astype(int)
model_placebo = smf.ols('returns ~ sentiment * PLACEBO + controls + FE',
                        data=pre_only).fit(cov_type='cluster', cov_kwds={'groups': pre_only['ticker']})
# Should find β_interaction ≈ 0 (no effect for fake treatment)
```

---

## 8. Data Policy & .gitignore

**Important**: Large data files and results are excluded from version control:

```gitignore
# Data files
data/raw/*.csv
data/processed/*.csv
data/results/*.csv
data/results/descriptive_stats/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
venv/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

Only configuration files, scripts, and documentation are committed.

---

## 9. Reproducibility

All analysis is fully reproducible:
- ✓ Fixed random seeds (PyTorch, NumPy where applicable)
- ✓ Pinned package versions in `requirements.txt`
- ✓ Git version control for all code
- ✓ Timestamped outputs with metadata
- ✓ Comprehensive documentation
- ✓ Clear pipeline stages with validation

---

## 10. Troubleshooting

### Common Issues

**Issue**: CUDA out of memory during Stage 1
```bash
# Reduce batch size in stage1_finbert_sentiment.py (line 30)
BATCH_SIZE = 16  # or 8 for very limited GPU memory
```

**Issue**: FinBERT model download fails
```python
# Pre-download the model before running pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

**Issue**: Few matches after merge with performance data
```bash
# Check ticker standardization
# Both datasets must have uppercase, trimmed tickers
sentiment['ticker'] = sentiment['ticker'].str.upper().str.strip()
performance['ticker'] = performance['ticker'].str.upper().str.strip()

# Check year formats
# Both should be integers
sentiment['year'] = sentiment['year'].astype(int)
performance['year'] = performance['year'].astype(int)
```

**Issue**: Regression coefficients not significant
```bash
# 1. Check sample size
print(f"Merged observations: {len(merged)}")  # Need >300 for sufficient power

# 2. Try alternative DVs
# FFO, occupancy rates, NAV growth instead of returns

# 3. Examine subsamples
# Data center REITs only, large REITs only, etc.

# 4. Check variable distributions
merged[['returns', 'sentiment', 'rating']].describe()
```

**Issue**: Pipeline stages fail with import errors
```bash
# Ensure you're running from the scripts/ directory
cd scripts/
python stage1_finbert_sentiment.py

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

**Issue**: Figures not generating or look corrupted
```bash
# Update matplotlib backend
pip install --upgrade matplotlib

# Or specify backend explicitly
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

**Issue**: Descriptive statistics show unexpected values
```bash
# Check data after each stage
import pandas as pd

# After Stage 1
df1 = pd.read_csv('data/processed/finbert_sentiment_scores.csv')
print(df1['sentiment_score'].describe())

# After Stage 2
df2 = pd.read_csv('data/processed/firm_year_sentiment.csv')
print(df2['sentiment_score_mean'].describe())

# Verify sentiment range is [-1, 1]
# Verify ratings range is [1, 5]
```

**Issue**: Out of memory errors on CPU
```bash
# Process in smaller batches
# Edit stage1_finbert_sentiment.py line 30:
BATCH_SIZE = 8  # Reduce from default 32
```

---

## 11. Expected Results

Based on literature (Huang et al. 2020; Green et al. 2019), expected findings:

### Baseline Model (RQ1)
- **Sentiment → Returns**: Positive coefficient (β ≈ 0.05 to 0.15)
- **Economic magnitude**: 1 SD increase in sentiment → 2-5% increase in annual returns

### Difference-in-Differences (RQ2)
- **ChatGPT moderator**: Positive interaction (β ≈ 0.03 to 0.08)
- **Interpretation**: Sentiment became 30-50% more predictive post-ChatGPT

### Triple-Difference (RQ3)
- **Tech-intensive × ChatGPT**: Positive three-way interaction (β ≈ 0.05 to 0.12)
- **Interpretation**: Effect concentrated in data center REITs



---

## 12. Variable Codebook - Key Variables

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `ticker` | string | - | REIT stock ticker symbol |
| `year` | integer | 2014-2025 | Calendar year |
| `sentiment` | float | [-1, 1] | **PRIMARY DV**: Mean FinBERT sentiment (positive - negative) |
| `sentiment_dispersion` | float | [0, ∞) | Within firm-year sentiment std dev (employee disagreement) |
| `sentiment_agreement` | float | [0, 1] | Proportion agreeing with majority sentiment |
| `rating` | float | [1, 5] | Mean Glassdoor rating |
| `review_count` | integer | [3, ∞) | Number of reviews (minimum 3 after filtering) |
| `POST_CHATGPT` | binary | {0, 1} | **KEY TREATMENT**: 1 if after Nov 30, 2022 |
| `tech_intensive` | binary | {0, 1} | 1 if Data Center/Telecom/Infrastructure |
| `datacenter_REIT` | binary | {0, 1} | 1 if Data Center only |
| `property_type` | string | - | Property type classification |

See `data/results/variable_codebook.md` for complete documentation (generated after Stage 4).

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@mastersthesis{niaz2025reit,
  author = {Niaz, Konain},
  title = {Employee Sentiment and REIT Performance: Evidence from FinBERT Analysis and the ChatGPT Era},
  school = {Rochester Institute of Technology},
  year = {2025},
  type = {Master's Thesis},
  note = {Advisors: Dr. Debanjana Dey, Dr. Travis Desell}
}
```

---

## References

### FinBERT Model
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.

### Employee Sentiment & Firm Performance
- Huang, K., Li, M., & Markov, S. (2020). What do employees know? Evidence from a social media platform. *The Accounting Review*, 95(2), 199-226.
- Green, T. C., et al. (2019). Crowdsourced employer reviews and stock returns. *Journal of Financial Economics*, 134(1), 236-251.

### Difference-in-Differences Methodology
- Angrist, J. D., & Pischke, J. S. (2008). *Mostly Harmless Econometrics*. Princeton University Press.

---

## Contact

**Konain Niaz**  
Email: kn4792@rit.edu  
GitHub: [@kn4792](https://github.com/kn4792)

---

## Acknowledgments

- **Dr. Debanjana Dey** (Finance Advisor)
- **Dr. Travis Desell** (Software Engineering Advisor)
- RIT Saunders College of Business
- RIT Golisano College of Computing and Information Sciences
- Anthropic's Claude (FinBERT pipeline development assistance)

---
