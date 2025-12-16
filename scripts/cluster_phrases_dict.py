import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# Read your custom phrase pairs from Excel
lexicon_df = pd.read_excel("data/processed/reit_ai_productivity_lexicon.xlsx")

# Display the first few rows to see the structure
print("Loaded phrases:")
print(lexicon_df.head())
print(f"\nTotal phrases: {len(lexicon_df)}")

# Assuming your Excel has a column with phrases - adjust column name as needed
# Common column names might be: 'phrase', 'term', 'Phrase', 'Task', etc.
# Let's check what columns exist
print(f"\nColumns in file: {lexicon_df.columns.tolist()}")

# Replace 'phrase' with your actual column name
# If you have multiple columns, you might want to combine them
# For example: lexicon_df['combined'] = lexicon_df['verb'] + ' ' + lexicon_df['object']
phrase_column = lexicon_df.columns[0]  # Using first column by default
vocab_phrases = lexicon_df[phrase_column].dropna().astype(str).str.strip().tolist()

# Remove empty strings and duplicates
vocab_phrases = [p for p in vocab_phrases if p]
vocab_phrases = list(dict.fromkeys(vocab_phrases))  # Remove duplicates while preserving order

print(f"\nUnique phrases to cluster: {len(vocab_phrases)}")

# Load sentence transformer model
print("\nLoading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings
print("Generating embeddings...")
phrase_embeddings = model.encode(vocab_phrases, show_progress_bar=True, convert_to_numpy=True)

# Dimensionality reduction with UMAP
print("\nReducing dimensions with UMAP...")
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    metric="cosine",
    random_state=42,
)

embeddings_umap = umap_model.fit_transform(phrase_embeddings)

# Clustering with HDBSCAN
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,      # Reduced from 20 since you might have fewer phrases
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(embeddings_umap)

# Create results dataframe
tasks_df = pd.DataFrame({
    "phrase": vocab_phrases,
    "cluster": cluster_labels
})

# Show cluster distribution
print(f"\nCluster distribution:")
print(tasks_df['cluster'].value_counts().sort_index())

# Separate clustered vs noise
clustered = tasks_df[tasks_df["cluster"] != -1].reset_index(drop=True)
noise = tasks_df[tasks_df["cluster"] == -1].reset_index(drop=True)

print(f"\nClustered phrases: {len(clustered)}")
print(f"Noise (unclustered): {len(noise)}")

# Save results
tasks_df.to_csv("data/processed/custom_clustered_phrases.csv", index=False)
print("\nAll results saved to 'data/processed/custom_clustered_phrases.csv'")

if len(clustered) > 0:
    clustered.to_csv("data/processed/custom_clustered_phrases_only.csv", index=False)
    print("Clustered phrases saved to 'data/processed/custom_clustered_phrases_only.csv'")

if len(noise) > 0:
    noise.to_csv("data/processed/custom_noise_phrases.csv", index=False)
    print("Noise phrases saved to 'data/processed/custom_noise_phrases.csv'")

# Display clusters
print("\n" + "="*80)
print("PHRASE CLUSTERS")
print("="*80)

cluster_groups = clustered.groupby("cluster")["phrase"].apply(list)

for cid, phrases in cluster_groups.items():
    print(f"\n=== Cluster {cid} ({len(phrases)} phrases) ===")
    for phrase in phrases[:30]:  # show first 30 phrases
        print(f"  - {phrase}")
    if len(phrases) > 30:
        print(f"  ... and {len(phrases) - 30} more")

# Show some noise examples if any
if len(noise) > 0:
    print(f"\n=== Unclustered/Noise ({len(noise)} phrases) ===")
    for phrase in noise["phrase"].head(20):
        print(f"  - {phrase}")
    if len(noise) > 20:
        print(f"  ... and {len(noise) - 20} more")