import pandas as pd

# Read the CSV file
df = pd.read_csv('data/raw/all_reviews.csv')

# Combine title, pros, and cons into one column
df['combined_text'] = (
    df['title'].fillna('').astype(str) + ' | ' + 
    df['pros'].fillna('').astype(str) + ' | ' + 
    df['cons'].fillna('').astype(str)
)

# Save the entire dataframe with the new combined column
df.to_csv('data/raw/combined_column.csv', index=False)

print("Combined column added and saved to 'data/raw/combined_column.csv'")
print(f"\nFirst few rows of combined text:")
print("-" * 80)
for i, text in enumerate(df['combined_text'].head(3), 1):
    print(f"{i}. {text[:200]}...")  # Show first 200 characters