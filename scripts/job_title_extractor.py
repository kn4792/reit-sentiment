import pandas as pd

# Read the CSV file
df = pd.read_csv('data/raw/all_reviews.csv')

# Extract unique job titles and remove empty/null values
unique_job_titles = df['job_title'].dropna().unique()

# Remove empty strings if any
unique_job_titles = [title for title in unique_job_titles if str(title).strip()]

# Sort the list
unique_job_titles = sorted(unique_job_titles)

# Print the results
print(f"Total unique job titles: {len(unique_job_titles)}\n")
print("Unique Job Titles:")
print("-" * 50)
for i, title in enumerate(unique_job_titles, 1):
    print(f"{i}. {title}")

# Save to CSV file
output_df = pd.DataFrame({
    'job_title': unique_job_titles
})

output_df.to_csv('unique_job_titles.csv', index=False)

print("\nResults saved to 'unique_job_titles.csv'")