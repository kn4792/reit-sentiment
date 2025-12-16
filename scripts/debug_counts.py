import os
import csv

csv_files = [
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_032441.csv',
    r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_085636.csv'
]

output_file = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\debug_report.txt'

with open(output_file, 'w', encoding='utf-8') as report:
    report.write(f"{'File':<60} | {'Raw Lines':<10} | {'CSV Records':<12}\n")
    report.write("-" * 90 + "\n")

    total_raw = 0
    total_records = 0

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        # Count raw lines
        with open(file_path, 'rb') as f:
            raw_lines = sum(1 for _ in f)
        
        # Count CSV records using csv module
        records = 0
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                records = sum(1 for _ in reader) - 1 # Subtract header
        except Exception as e:
            report.write(f"Error reading {file_name}: {e}\n")
            records = -1

        report.write(f"{file_name:<60} | {raw_lines:<10} | {records:<12}\n")
        
        total_raw += raw_lines
        total_records += records

    report.write("-" * 90 + "\n")
    report.write(f"{'Total':<60} | {total_raw:<10} | {total_records:<12}\n")

print(f"Debug report written to {output_file}")
