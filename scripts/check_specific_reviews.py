import pandas as pd
import os

# Configuration
csv_files = [
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251203_210354.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_032441.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_085636.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_154433.csv'
]

output_file = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\specific_report.txt'

def analyze_files(csv_files, output_path):
    all_reviews = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Specific REIT Reviews Analysis Report\n")
        f.write("=====================================\n\n")
        
        f.write(f"{'File':<60} | {'Reviews':<10} | {'Unique Tickers':<15}\n")
        f.write("-" * 90 + "\n")

        for file_path in csv_files:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    file_name = os.path.basename(file_path)
                    num_reviews = len(df)
                    unique_tickers = df['ticker'].nunique()
                    
                    f.write(f"{file_name:<60} | {num_reviews:<10} | {unique_tickers:<15}\n")
                    
                    all_reviews.append(df)
                else:
                    f.write(f"{os.path.basename(file_path):<60} | {'NOT FOUND':<10} | {'N/A':<15}\n")
                
            except Exception as e:
                f.write(f"Error processing {os.path.basename(file_path)}: {e}\n")

        f.write("-" * 90 + "\n")
        
        # Combine all data
        if all_reviews:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            total_reviews = len(combined_df)
            total_unique_tickers = combined_df['ticker'].nunique()
            
            f.write(f"\nTotal Reviews across all files: {total_reviews}\n")
            f.write(f"Total Unique Tickers across all files: {total_unique_tickers}\n")
            
            # List all unique tickers found
            unique_tickers_list = sorted(combined_df['ticker'].unique().tolist())
            f.write(f"\nUnique Tickers Found ({len(unique_tickers_list)}): {unique_tickers_list}\n")

if __name__ == "__main__":
    analyze_files(csv_files, output_file)
    print(f"Report written to {output_file}")
