import pandas as pd
import spacy
from collections import defaultdict, Counter

from sentence_transformers import SentenceTransformer
import umap
import hdbscan


# Read the CSV file with combined column
df = pd.read_csv("data/raw/combined_column.csv")

# Use the combined_text column instead of Reviews
df = df.dropna(subset=["combined_text"])
df["text"] = df["combined_text"].astype(str).str.strip()
df = df[df["text"].str.len() > 0].reset_index(drop=True)


nlp = spacy.load("en_core_web_sm", disable=["ner"])  # speed

def extract_task_phrases(text):
    doc = nlp(text.lower())
    phrases = []

    # 1) Verb–object / verb–prep–object patterns
    for token in doc:
        # direct object: "handle complaints", "write reports"
        if token.pos_ == "VERB":
            dobj = [child for child in token.children if child.dep_ == "dobj"]
            pobj = []
            # prepositional object: "talk to tenants", "coordinate with vendors"
            for child in token.children:
                if child.dep_ == "prep":
                    pobj.extend([gc for gc in child.children if gc.dep_ == "pobj"])

            # build phrases
            for obj in dobj + pobj:
                phrase = f"{token.lemma_} {obj.lemma_}"
                phrases.append(phrase)

    # 2) Noun chunks that look like tasks/duties
    for chunk in doc.noun_chunks:
        txt = chunk.lemma_.strip()
        # filter obvious non-task things a bit
        if len(txt.split()) <= 5 and not txt.isdigit():
            phrases.append(txt)

    # de-duplicate within review
    phrases = list(dict.fromkeys(phrases))
    return phrases

df["task_phrases"] = df["text"].apply(extract_task_phrases)
df[["text", "task_phrases"]].head(10)



all_phrases = [phrase for phrases in df["task_phrases"] for phrase in phrases]
phrase_counts = Counter(all_phrases)

# keep phrases that appear at least N times (tune this, e.g., 10)
MIN_FREQ = 10
vocab_phrases = [p for p, c in phrase_counts.items() if c >= MIN_FREQ]

len(vocab_phrases)



model = SentenceTransformer("all-mpnet-base-v2")  # good general model

phrase_embeddings = model.encode(vocab_phrases, show_progress_bar=True, convert_to_numpy=True)



# Dimensionality reduction
umap_model = umap.UMAP(
    n_neighbors=15,
    n_components=10,
    metric="cosine",
    random_state=42,
)

embeddings_umap = umap_model.fit_transform(phrase_embeddings)

# Clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,      # tune based on dataset size
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(embeddings_umap)

tasks_df = pd.DataFrame({
    "phrase": vocab_phrases,
    "cluster": cluster_labels
})

# ignore noise cluster (-1) for now
tasks_df = tasks_df[tasks_df["cluster"] != -1].reset_index(drop=True)

tasks_df.head()


tasks_df.to_csv("clustered_task_phrases.csv", index=False)

cluster_groups = tasks_df.groupby("cluster")["phrase"].apply(list)

for cid, phrases in cluster_groups.items():
    print(f"\n=== Cluster {cid} ===")
    print(phrases[:30])  # show first 30 phrases



