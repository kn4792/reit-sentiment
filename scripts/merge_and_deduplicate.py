import pandas as pd
import os

# Configuration
csv_files = [
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251203_210354.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_032441.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_085636.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_154433.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_162620.csv'
]

output_csv = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reviews.csv'
report_file = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\merge_report.txt'

def merge_and_deduplicate(csv_files, output_csv, report_path):
    all_reviews = []
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Merge and Deduplication Report\n")
        f.write("==============================\n\n")
        
        f.write(f"{'File':<60} | {'Reviews':<10}\n")
        f.write("-" * 75 + "\n")

        # Read all files
        for file_path in csv_files:
            try:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    file_name = os.path.basename(file_path)
                    num_reviews = len(df)
                    f.write(f"{file_name:<60} | {num_reviews:<10}\n")
                    all_reviews.append(df)
                else:
                    f.write(f"{os.path.basename(file_path):<60} | {'NOT FOUND':<10}\n")
            except Exception as e:
                f.write(f"Error processing {os.path.basename(file_path)}: {e}\n")

        f.write("-" * 75 + "\n")
        
        if all_reviews:
            combined_df = pd.concat(all_reviews, ignore_index=True)
            total_initial = len(combined_df)
            
            f.write(f"\nTotal Initial Reviews: {total_initial}\n")
            
            # Deduplicate
            subset_cols = ['title', 'rating', 'date', 'job_title', 'pros', 'cons', 'ticker']
            # Keep first occurrence
            deduplicated_df = combined_df.drop_duplicates(subset=subset_cols, keep='first')
            
            total_final = len(deduplicated_df)
            duplicates_removed = total_initial - total_final
            
            f.write(f"Duplicates Removed: {duplicates_removed}\n")
            f.write(f"Final Total Reviews: {total_final}\n")
            
            # Save merged file
            deduplicated_df.to_csv(output_csv, index=False)
            f.write(f"\nMerged file saved to: {output_csv}\n")
            
            # Final stats
            unique_tickers = deduplicated_df['ticker'].nunique()
            f.write(f"Total Unique Tickers in Final Dataset: {unique_tickers}\n")
            
        else:
            f.write("\nNo data to merge.\n")

if __name__ == "__main__":
    merge_and_deduplicate(csv_files, output_csv, report_file)
    print(f"Merge report written to {report_file}")
