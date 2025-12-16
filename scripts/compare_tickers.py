import pandas as pd
import json
import os

# Configuration
json_path = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\config\reit_companies.json'
csv_files = [
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251203_210354.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_032441.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_085636.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_154433.csv'
]

output_file = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\ticker_comparison_report.txt'

def load_expected_tickers(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return set(company['ticker'] for company in data['companies'])

def analyze_tickers(csv_files, json_path, output_path):
    expected_tickers = load_expected_tickers(json_path)
    found_tickers = set()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Ticker Comparison Report\n")
        f.write("========================\n\n")
        
        # Collect tickers from CSVs
        for file_path in csv_files:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    if 'ticker' in df.columns:
                        file_tickers = set(df['ticker'].unique())
                        found_tickers.update(file_tickers)
                    else:
                        f.write(f"WARNING: No 'ticker' column in {os.path.basename(file_path)}\n")
                else:
                    f.write(f"WARNING: File not found: {os.path.basename(file_path)}\n")
            except Exception as e:
                f.write(f"Error reading {os.path.basename(file_path)}: {e}\n")

        # Analysis
        unexpected_tickers = found_tickers - expected_tickers
        missing_coverage = expected_tickers - found_tickers
        
        f.write(f"Total Expected Tickers (JSON): {len(expected_tickers)}\n")
        f.write(f"Total Found Tickers (CSVs): {len(found_tickers)}\n\n")
        
        f.write("1. Tickers in CSVs but NOT in JSON (Unexpected):\n")
        if unexpected_tickers:
            f.write(f"   Count: {len(unexpected_tickers)}\n")
            f.write(f"   Tickers: {sorted(list(unexpected_tickers))}\n")
        else:
            f.write("   None. All found tickers are valid according to JSON.\n")
            
        f.write("\n2. Tickers in JSON but NOT in CSVs (Missing Coverage):\n")
        if missing_coverage:
            f.write(f"   Count: {len(missing_coverage)}\n")
            f.write(f"   Tickers: {sorted(list(missing_coverage))}\n")
        else:
            f.write("   None. Full coverage achieved.\n")

if __name__ == "__main__":
    analyze_tickers(csv_files, json_path, output_file)
    print(f"Comparison report written to {output_file}")
