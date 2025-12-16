import pandas as pd
import os

file_path = r'c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\data\raw\all_reit_reviews_20251204_024439.csv'

try:
    df = pd.read_csv(file_path, nrows=0)
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error reading CSV: {e}")
