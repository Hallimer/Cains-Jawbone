from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import spacy

# 1. Load and split pages
# Load raw unformatted Cain's Jawbone text
with open('data/Cains_Jawbone_Unformatted.txt', encoding='utf-8') as f:
    raw_text = f.read()

# Split into pages based on ________________
split_pages = [page.strip() for page in raw_text.strip().split("________________") if page.strip()]
assert len(split_pages) == 100, f"Expected 100 pages, got {len(split_pages)}"

# 2. Process and Store Results
classified = []
for i, page in enumerate(split_pages, start=1):
    classified.append({
        "Page": i,
        "Preview": page[:100].replace("\n", " ") + "..."
    })

df = pd.DataFrame(classified)
df.head(10)

# 3. Named Entity Recognition (NER) using spaCy
# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "GPE", "ORG", "NORP")]

df["Entities"] = split_pages
df["Entities"] = df["Entities"].apply(extract_entities)


# 4. K-Means Clustering
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(split_pages)

k = 6  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# 5. Save Results
df.to_csv("results/cains_jawbone_analysis.csv", index=False)