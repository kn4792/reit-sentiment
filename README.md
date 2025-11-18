# REIT Sentiment & Generative AI Impact Study

**Author:** Konain Niaz (kn4792@rit.edu)  
**Advisors:** Dr. Debanjana Dey, Dr. Travis Desell  
**Institution:** Rochester Institute of Technology  

---

## 1. Project Overview

This project analyzes the relationship between Generative AI adoption, employee sentiment, and firm performance in US REITs (Real Estate Investment Trusts) using large-scale Glassdoor review scraping, NLP-based sentiment analysis (FinBERT). The pipeline covers scraping, cleaning, sentiment computation, keyword extraction, and testing/visualization.

Then next steps of this project will be to create a measure of AI related productivity using MNIR, given the sentiment scores and extracted keywords. The measure will then be checked using a difference in difference approach by looking at the timeline of the measure before and after AI rollout, somewhere around November 30th 2022 (ChatGPT came to market). 

---

## 2. System Requirements

### Required Software
- **Python**: 3.9 or higher
- **Google Chrome**: Latest version (for web scraping)
- **ChromeDriver**: Matching your Chrome version
- **Recommended**: virtualenv or conda for environment management

### Hardware Recommendations
- **RAM**: 16GB minimum (32GB recommended for large-scale processing)
- **Storage**: 10GB available space
- **GPU**: CUDA-compatible GPU recommended for FinBERT (optional)

### Account Requirements
- Glassdoor Username and Password

---

## 3. Installation

### Clone Repository & Set Up Environment
```bash
# Clone repository
git clone https://github.com/kn4792/reit-sentiment.git
cd reit-sentiment       # or the exact folder address within your computer

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
pandas
numpy
requests
selenium
finbert-embedding
transformers
scikit-learn
matplotlib
seaborn
pytest
nltk
```

### Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 4. Directory Structure

```
reit-sentiment-analysis/
│
├── config/
│   ├── reit_companies.json      # List of 127 REITs with Glassdoor URLs
│   └── schema.sql                # MySQL database schema (optional)
│
├── data/
│   ├── raw/                      # Raw scraped data (.gitignored)
│   ├── processed/                # Cleaned data, train/val/test splits
│   ├── results/                  # Analysis results, models, plots
│   │   ├── exploration/          # Exploratory analysis outputs
│   │   ├── plots/                # Generated visualizations
│   │   └── tables/               # Generated tables and reports
│   └── sample/                   # Sample data for testing
│
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   └── glassdoor_scraper.py  # Web scraping with Selenium
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py       # Text preprocessing pipeline
│   │   └── data_validator.py     # Data quality checks
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sentiment_analyzer.py # FinBERT sentiment analysis
│   │   ├── keyword_extractor.py  # TF-IDF + keyword extraction
│   │   └── ai_productivity.py    # AI productivity measurement
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── panel_regression.py   # Fixed effects models
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py           # Matplotlib/Seaborn visualizations
│
├── scripts/
│   ├── scrape_reviews.py         # Step 1: Data collection
│   ├── clean_data.py             # Step 2: Data cleaning
│   ├── split_dataset.py          # Step 3: Train/val/test split
│   ├── sentiment_analysis.py     # Step 4: Sentiment scoring
│   ├── keyword_extraction.py     # Step 5: Keyword analysis
│   ├── train_model.py            # Step 6: Model training & evaluation
│   ├── explore_data.py           # Step 7: Exploratory analysis
│   ├── generate_plots.py         # Step 8: Results visualization
│   └── generate_tables.py        # Step 9: Table generation
│
├── tests/
│   ├── __init__.py
│   │
│   ├── Phase 1: Data Preparation Tests
│   ├── test_data_preparation.py  # Integration test for Phase 1
│   ├── test_scraper.py           # Unit tests for scraping
│   ├── test_cleaner.py           # Unit tests for cleaning
│   │
│   ├── Phase 2: Model Training Tests
│   ├── test_model_training.py    # Integration test for Phase 2
│   ├── test_sentiment.py         # Unit tests for sentiment analysis
│   │
│   ├── Phase 3: Exploration Tests
│   ├── test_exploration.py       # Tests for data exploration
│   │
│   ├── Phase 4: Visualization Tests
│   ├── test_visualization.py     # Integration test for Phase 4
│   │
│   └── conftest.py               # Pytest configuration and fixtures
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
│
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 5. Pipeline Execution

#### Phase 1.1: Initiate Chrome in debugging mode manually, and log in to glassdoor.com

Start Chrome in debugging mode first:

```bash
# Windows
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\selenium\ChromeProfile"

# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir="~/selenium/ChromeProfile"

# Linux
google-chrome --remote-debugging-port=9222
```

In the newly opened Chrome window, go to glassdoor.com and log in with your credentials. 

### Phase 1.2: Scrape Single Company Reviews by Ticker

```bash
# Phase 1: Data Collection & Preparation
python scripts/scrape_reviews.py --company PLD --max-reviews 100
```

**Note**: Manual Glassdoor login required. You need to press enter once the log in is successful. You will see the scraper go through the pages. Output: 1 CSV per REIT + combined dataset.

Run tests to verify data was scraped correctly:
```bash
pytest tests/test_scraper.py -v          # Verify scraping logic
```

#### Phase 1.3: Data Cleaning & Preprocessing

```bash
python scripts/clean_data.py --input data/raw/PLD_reviews.csv --output data/processed/
```

**Cleaning steps:**
1. Remove HTML entities and special characters
2. Tokenize text using NLTK
3. Remove stopwords (including domain-specific: "reit", "company")
4. Apply Porter stemming
5. Normalize dates to YYYY-MM-DD format
6. Handle missing values
7. Remove outliers (>5 standard deviations)

#### Phase 1.4: Dataset Splitting
```bash
python scripts/split_dataset.py --input data/processed/cleaned_PLD_reviews.csv --outdir data/processed/PLD/
```
Creates `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.


Run tests to verify the data was prepared (cleaned and split) correctly:

```bash
pytest tests/test_data_preparation.py -v
```

### Phase 2: Model Training
#### Phase 2.1: Sentiment Analysis & Keyword Extraction

```bash
# Run sentiment analysis on splits
python scripts/sentiment_analysis.py --input data/processed/PLD/train.csv --output data/results/PLD/sentiments_train
python scripts/sentiment_analysis.py --input data/processed/PLD/val.csv --output data/results/PLD/sentiments_val
python scripts/sentiment_analysis.py --input data/processed/PLD/test.csv --output data/results/PLD/sentiments_test
```

Test to see that sentiment analysis was done correctly:

```bash
pytest tests/test_sentiment.py -v
```


##### Extract keywords
```bash
python scripts/keyword_extraction.py --input data/processed/PLD/train.csv --output data/results/PLD/keywords_train
```
Test to verify the keywords were extracted correctly:
```bash
pytest tests/test_keyword_extraction.py -v
```



#### Phase 2.2: Model Training
```bash
# Train on default parameters
python scripts/train_model.py --train data/results/PLD/sentiments_train --val data/results/PLD/sentiments_val --test data/results/PLD/sentiments_test --output data/results/PLD/model_results
```

Verify model training and evaluation:
```bash
pytest tests/test_model_training.py -v
```


### Phase 3: Exploration

#### Phase 3.1: Exploratory Data Analysis
```bash
python scripts/explore_data.py --input data/processed/cleaned_PLD_reviews.csv --output-dir data/results/PLD/exploration
```

Test to check exploration worked correctly:
```bash
pytest tests/test_exploration.py -v
```

### Phase 4: Visualization and Results

#### Phase 4.1: Generate Results Visualization
```bash
#generate all plots
python scripts/generate_plots.py --input data/results/PLD/model_results --output-dir data/results/PLD/plots
```

Test to verify results visualization:
```bash
pytest tests/test_visualization.py -v
```


---

## 6. Data Policy & .gitignore

**Important**: Large and intermediate data files are excluded from version control:

```gitignore
data/raw/*.csv
data/raw/*.json
data/processed/*.csv
*.parquet
__pycache__/
*.pyc
.env
```

Only sample data and configuration files are committed.

---

## 7. Reproducibility

All analysis is fully reproducible:
- ✓ Fixed random seeds (`random.seed(42)`, `np.random.seed(42)`)
- ✓ Pinned package versions in `requirements.txt`
- ✓ Git version control for all code
- ✓ Timestamped data outputs
- ✓ Comprehensive logging
- ✓ Train/validation/test splitting

---

## 8. Troubleshooting

### Common Issues

**Issue**: Chrome debug mode not connecting
```bash
# Kill existing Chrome processes and restart
taskkill /F /IM chrome.exe  # Windows
pkill -9 chrome              # macOS/Linux
```

**Issue**: FinBERT out of memory
```bash
# Reduce batch size
python scripts/sentiment_analysis.py --batch-size 8
```

**Issue**: Tests failing due to missing sample data
```bash
# Ensure sample data exists
ls data/sample/

# If missing, create sample data from your processed data
python scripts/create_sample_data.py --input data/processed/cleaned_all_reviews.csv --output data/sample/ --size 100
```

**Issue**: Import errors when running tests
```bash
# Install package in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

**Issue**: Pytest not discovering tests
```bash
# Check pytest configuration
pytest --collect-only

# Run from project root directory
cd /path/to/reit-sentiment-analysis
pytest tests/
```

**Issue**: Coverage report not including all files
```bash
# Ensure .coveragerc or pyproject.toml is configured
# Run with explicit source paths
pytest tests/ --cov=src --cov=scripts --cov-report=html
```

**Issue**: Tests timing out or running too slowly
```bash
# Run only fast unit tests
pytest tests/ -v -m "not slow"

# Increase timeout for specific tests
pytest tests/ --timeout=300

# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto
```

**Issue**: Selenium WebDriver version mismatch
```bash
# Check Chrome version
google-chrome --version  # Linux
# Update ChromeDriver to match

# Or use webdriver-manager (auto-downloads correct version)
pip install webdriver-manager
```

---

## 9. Citation

If you use this code or data, please cite:

```bibtex
@mastersthesis{niaz2025reit,
  author = {Niaz, Konain},
  title = {AI, Management Perception, and Corporate Culture: Employee Sentiment and Its Effect on REIT Transparency and Performance},
  school = {Rochester Institute of Technology},
  year = {2025},
  type = {Master's Thesis}
}
```

---

## 10. Contact

**Konain Niaz**  
Email: kn4792@rit.edu  
GitHub: [@kn4792](https://github.com/kn4792)

---

## 11. Acknowledgments

- Dr. Debanjana Dey (Advisor)
- Dr. Travis Desell (Advisor)
- RIT Data Science Program
- MatthewChatham's glassdoor-review-scraper (base scraping code)