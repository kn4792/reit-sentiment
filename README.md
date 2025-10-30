# REIT Sentiment Analysis Using Employee Reviews

**Authors:** Konain Niaz, Travis Desell, Debanjana Dey  
**Institution:** Rochester Institute of Technology  
**Course:** DSCI 601

## Project Overview

This project performs sentiment analysis on Glassdoor employee reviews to analyze and forecast Real Estate Investment Trust (REIT) asset pricing and returns. We integrate employee sentiment signals with traditional REIT factor models using natural language processing and deep learning techniques.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Analysis and Visualization](#analysis-and-visualization)
- [Running Tests](#running-tests)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)

---

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 8GB RAM (16GB recommended for model training)
- **Storage**: At least 5GB free space for data and models

### Python Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- **Data Collection**: selenium, beautifulsoup4, playwright
- **NLP/ML**: torch, transformers (FinBERT), scikit-learn, nltk, spacy
- **Financial Analysis**: pandas, numpy, statsmodels, yfinance
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, pytest-cov

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/reit-sentiment-analysis.git
cd reit-sentiment-analysis
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords vader_lexicon

# Download spaCy model
python -m spacy download en_core_web_sm

# Install Playwright browsers (for web scraping)
playwright install chromium
```

### 4. Configure Credentials
Create a `config/secret.json` file with your Glassdoor credentials:
```json
{
  "glassdoor_username": "your_email@example.com",
  "glassdoor_password": "your_password"
}
```

**Important:** Never commit `secret.json` to version control. It's listed in `.gitignore`.

### 5. Verify Installation
```bash
# Run installation test
pytest tests/test_installation.py -v
```

---

## Project Structure

```
reit-sentiment-analysis/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── config/                        # Configuration files
│   ├── config.yaml               # Project settings
│   ├── reit_companies.json       # List of REIT companies to scrape
│   └── secret.json               # Credentials (not in git)
│
├── data/                         # Data directory
│   ├── raw/                      # Raw scraped data
│   ├── processed/                # Cleaned and processed data
│   └── results/                  # Analysis results
│
├── src/                          # Source code
│   ├── data_collection/          # Web scraping modules
│   ├── preprocessing/            # Data cleaning and feature engineering
│   ├── models/                   # ML/NLP models
│   ├── analysis/                 # Correlation and factor analysis
│   └── visualization/            # Plotting and visualization
│
├── scripts/                      # Executable scripts
│   ├── download_data.py          # Data collection script
│   ├── preprocess_data.py        # Preprocessing script
│   ├── train_model.py            # Model training script
│   ├── run_analysis.py           # Analysis script
│   └── generate_plots.py         # Visualization script
│
├── tests/                        # Unit tests
│   ├── test_data_collection.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_visualization.py
│
└── notebooks/                    # Jupyter notebooks for exploration
    ├── exploratory_analysis.ipynb
    └── results_presentation.ipynb
```

---

## Data Collection

### Step 1: Configure REIT Companies

Edit `config/reit_companies.json` to specify which REITs to scrape:
```json
{
  "companies": [
    {
      "name": "Prologis",
      "glassdoor_url": "https://www.glassdoor.com/Reviews/Prologis-Reviews-E8449.htm",
      "ticker": "PLD"
    },
    {
      "name": "American Tower",
      "glassdoor_url": "https://www.glassdoor.com/Reviews/American-Tower-Reviews-E14129.htm",
      "ticker": "AMT"
    }
  ]
}
```

### Step 2: Scrape Glassdoor Reviews

```bash
# Scrape all companies (default: 500 reviews per company)
python scripts/download_data.py

# Scrape specific company
python scripts/download_data.py --company "Prologis" --max-reviews 1000

# Scrape with custom date range
python scripts/download_data.py --start-date "2020-01-01" --end-date "2023-12-31"
```

**Output:** Raw review data saved to `data/raw/glassdoor_reviews.csv`

### Step 3: Download REIT Financial Data

```bash
# Download REIT returns and factor data
python scripts/download_data.py --financial-only

# Download specific date range
python scripts/download_data.py --financial-only --start-date "2020-01-01"
```

**Output:** Financial data saved to `data/raw/reit_returns.csv`

### Step 4: Verify Data Collection

```bash
# Run data collection tests
pytest tests/test_data_collection.py -v

# Check data integrity
python scripts/download_data.py --verify
```

**Expected Output:**
```
✓ Glassdoor reviews: 5,432 reviews from 12 companies
✓ Date range: 2018-01-15 to 2023-12-20
✓ REIT returns: 145 months of data for 12 tickers
✓ All required columns present
```

---

## Data Preprocessing

### Step 1: Clean and Process Text Data

```bash
# Run full preprocessing pipeline
python scripts/preprocess_data.py

# Process only reviews (skip financial data)
python scripts/preprocess_data.py --reviews-only

# Custom text cleaning options
python scripts/preprocess_data.py --lowercase --remove-stopwords --lemmatize
```

**Processing Steps:**
1. Text normalization (lowercase, remove special characters)
2. Tokenization
3. Stop word removal
4. Lemmatization/stemming
5. Sentiment feature extraction
6. TF-IDF vectorization

**Output:** Processed data saved to `data/processed/`

### Step 2: Feature Engineering

```bash
# Generate sentiment features
python scripts/preprocess_data.py --feature-engineering

# Options:
#   --tfidf           Generate TF-IDF features
#   --word2vec        Generate word embeddings
#   --finbert         Use FinBERT embeddings (slow, requires GPU)
```

### Step 3: Align Reviews with Financial Data

```bash
# Merge sentiment scores with REIT returns by date
python scripts/preprocess_data.py --merge-financial
```

**Output:** `data/processed/merged_data.csv` with columns:
- `date`, `company`, `ticker`
- `sentiment_score`, `pros_sentiment`, `cons_sentiment`
- `return`, `excess_return`, `volatility`
- Factor loadings: `size`, `value`, `momentum`, `quality`, `low_vol`, `reversal`

### Step 4: Verify Preprocessing

```bash
# Run preprocessing tests
pytest tests/test_preprocessing.py -v
```

---

## Model Training

### Option 1: FinBERT Sentiment Model (Recommended)

```bash
# Fine-tune FinBERT on REIT reviews
python scripts/train_model.py --model finbert --epochs 5

# With GPU acceleration
python scripts/train_model.py --model finbert --epochs 5 --device cuda

# Resume training from checkpoint
python scripts/train_model.py --model finbert --resume --checkpoint models/finbert_epoch3.pt
```

### Option 2: LSTM Sentiment Model

```bash
# Train LSTM model
python scripts/train_model.py --model lstm --epochs 10 --hidden-size 128

# With hyperparameter tuning
python scripts/train_model.py --model lstm --tune-hyperparams
```

### Option 3: Traditional ML Models

```bash
# Train SVM classifier
python scripts/train_model.py --model svm

# Train logistic regression
python scripts/train_model.py --model logistic
```

### Training Options

```bash
python scripts/train_model.py \
  --model finbert \
  --train-split 0.7 \
  --val-split 0.15 \
  --test-split 0.15 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --epochs 5 \
  --early-stopping \
  --save-best-only
```

**Output:** Trained models saved to `models/`

### Verify Model Training

```bash
# Run model tests with sample data
pytest tests/test_models.py -v

# Evaluate model performance
python scripts/train_model.py --evaluate --model-path models/finbert_best.pt
```

**Expected Output:**
```
Model Evaluation Results:
  Accuracy: 0.847
  Precision: 0.823
  Recall: 0.856
  F1 Score: 0.839
  ROC-AUC: 0.912
```

---

## Analysis and Visualization

### Step 1: Correlation Analysis

```bash
# Compute correlations between sentiment and returns
python scripts/run_analysis.py --correlation

# Test Granger causality
python scripts/run_analysis.py --granger-causality --max-lag 12
```

### Step 2: Factor Model Analysis

```bash
# Run Fama-French style regression with sentiment
python scripts/run_analysis.py --factor-model

# GARCH model for volatility analysis
python scripts/run_analysis.py --garch-model
```

### Step 3: Backtesting

```bash
# Backtest trading strategy based on sentiment
python scripts/run_analysis.py --backtest --strategy sentiment-momentum

# Custom parameters
python scripts/run_analysis.py --backtest \
  --strategy sentiment-momentum \
  --lookback 30 \
  --rebalance-freq monthly \
  --transaction-cost 0.001
```

### Step 4: Generate Visualizations

```bash
# Generate all plots
python scripts/generate_plots.py

# Specific plot types
python scripts/generate_plots.py --plot sentiment-timeseries
python scripts/generate_plots.py --plot correlation-heatmap
python scripts/generate_plots.py --plot factor-loadings
python scripts/generate_plots.py --plot backtest-returns
```

**Output:** Plots saved to `data/results/figures/`

### Step 5: Verify Analysis

```bash
# Run analysis tests
pytest tests/test_analysis.py -v
pytest tests/test_visualization.py -v
```

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Modules
```bash
# Data collection tests
pytest tests/test_data_collection.py -v

# Preprocessing tests
pytest tests/test_preprocessing.py -v

# Model tests
pytest tests/test_models.py -v

# Visualization tests
pytest tests/test_visualization.py -v
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run Quick Smoke Tests (Small Data)
```bash
pytest tests/ -m smoke
```

---

## Usage Examples

### Example 1: Complete Pipeline

```bash
# 1. Download data
python scripts/download_data.py --max-reviews 1000

# 2. Preprocess
python scripts/preprocess_data.py --feature-engineering

# 3. Train model
python scripts/train_model.py --model finbert --epochs 3

# 4. Run analysis
python scripts/run_analysis.py --correlation --factor-model

# 5. Generate plots
python scripts/generate_plots.py
```

### Example 2: Using as Python Module

```python
from src.data_collection import GlassdoorScraper
from src.preprocessing import TextCleaner
from src.models import FinBERTSentiment

# Scrape reviews
scraper = GlassdoorScraper(credentials_path='config/secret.json')
reviews = scraper.scrape_company('Prologis', max_reviews=500)

# Preprocess text
cleaner = TextCleaner()
cleaned_reviews = cleaner.clean(reviews['pros'])

# Analyze sentiment
sentiment_model = FinBERTSentiment()
sentiment_scores = sentiment_model.predict(cleaned_reviews)
```

### Example 3: Custom Analysis in Jupyter

See `notebooks/exploratory_analysis.ipynb` for interactive examples.

---

## Contributing

### Code Review Process

1. **Create feature branch:** `git checkout -b feature/your-feature`
2. **Write tests:** Add tests in `tests/` directory
3. **Run tests locally:** `pytest tests/ -v`
4. **Format code:** `black src/ tests/ scripts/`
5. **Check style:** `pylint src/`
6. **Commit changes:** `git commit -m "Description"`
7. **Push branch:** `git push origin feature/your-feature`
8. **Create Pull Request:** Request code review from team member

### Code Style Guidelines

- Follow PEP 8 style guide
- Use Black formatter (max line length: 88)
- Write docstrings for all functions/classes (Google style)
- Maintain test coverage above 80%
- Add type hints where appropriate

---

## Troubleshooting

### Common Issues

**Issue:** Selenium WebDriver not found
```bash
# Solution: Install ChromeDriver
playwright install chromium
```

**Issue:** Out of memory during model training
```bash
# Solution: Reduce batch size
python scripts/train_model.py --model finbert --batch-size 8
```

**Issue:** CUDA out of memory (GPU)
```bash
# Solution: Use CPU or reduce batch size
python scripts/train_model.py --model finbert --device cpu
```

**Issue:** Glassdoor blocking requests
```bash
# Solution: Add delays and rotate user agents
python scripts/download_data.py --delay 5 --rotate-agents
```

---


---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---



**Project Repository:** https://github.com/your-username/reit-sentiment-analysis