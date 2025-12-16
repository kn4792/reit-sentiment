import pandas as pd
import json
import os

# Configuration
json_path = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\config\reit_companies.json'
csv_files = [
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_032441.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_085636.csv'
]

output_file = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\report.txt'

def load_valid_tickers(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return set(company['ticker'] for company in data['companies'])

def analyze_files(csv_files, valid_tickers, output_path):
    all_reviews = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("REIT Reviews Analysis Report\n")
        f.write("============================\n\n")
        
        f.write(f"Found {len(valid_tickers)} valid tickers in JSON.\n\n")
        
        f.write(f"{'File':<60} | {'Reviews':<10} | {'Unique Tickers':<15}\n")
        f.write("-" * 90 + "\n")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                file_name = os.path.basename(file_path)
                num_reviews = len(df)
                unique_tickers = df['ticker'].nunique()
                found_tickers = set(df['ticker'].unique())
                
                f.write(f"{file_name:<60} | {num_reviews:<10} | {unique_tickers:<15}\n")
                
                # Check for invalid tickers
                invalid_tickers = found_tickers - valid_tickers
                if invalid_tickers:
                    f.write(f"  WARNING: Found {len(invalid_tickers)} tickers not in JSON: {invalid_tickers}\n")
                
                all_reviews.append(df)
                
            except Exception as e:
                f.write(f"Error processing {file_path}: {e}\n")

        f.write("-" * 90 + "\n")
        
        # Combine all data
        if all_reviews:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            total_reviews = len(combined_df)
            total_unique_tickers = combined_df['ticker'].nunique()
            
            f.write(f"\nTotal Reviews across all files: {total_reviews}\n")
            f.write(f"Total Unique Tickers across all files: {total_unique_tickers}\n")
            
            found_tickers_all = set(combined_df['ticker'].unique())
            missing_tickers_all = valid_tickers - found_tickers_all
            if missing_tickers_all:
                f.write(f"Tickers from JSON NOT found in any file ({len(missing_tickers_all)}): {sorted(list(missing_tickers_all))}\n")
            else:
                f.write("All tickers from JSON are present in the data.\n")

            # Check for duplicates
            subset_cols = ['title', 'rating', 'date', 'job_title', 'pros', 'cons', 'ticker']
            duplicates = combined_df.duplicated(subset=subset_cols, keep='first')
            num_duplicates = duplicates.sum()
            
            f.write(f"\nDuplicate Reviews (based on content): {num_duplicates}\n")
            
            if num_duplicates > 0:
                f.write("Example duplicates:\n")
                f.write(combined_df[duplicates].head(3)[['ticker', 'date', 'title']].to_string() + "\n")

if __name__ == "__main__":
    valid_tickers = load_valid_tickers(json_path)
    analyze_files(csv_files, valid_tickers, output_file)
    print(f"Report written to {output_file}")
