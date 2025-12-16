import csv
import os

file_path = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv'

print(f"Scanning {os.path.basename(file_path)} for multi-line records...")

with open(file_path, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    prev_line_num = 1
    
    for i, row in enumerate(reader, start=1):
        curr_line_num = reader.line_num
        lines_spanned = curr_line_num - prev_line_num
        
        if lines_spanned > 1:
            print(f"\nFound multi-line record at Record #{i}")
            print(f"Raw File Lines: {prev_line_num + 1} to {curr_line_num}")
            print(f"Spans {lines_spanned} lines.")
            
            # Find which field has newlines
            for col_idx, value in enumerate(row):
                if '\n' in value:
                    col_name = header[col_idx] if col_idx < len(header) else f"Col {col_idx}"
                    print(f"Field '{col_name}' contains newlines:")
                    print("-" * 20)
                    print(value)
                    print("-" * 20)
            
            break # Stop after finding the first one
            
        prev_line_num = curr_line_num
