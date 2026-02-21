import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# ENCODER: lightweight but powerful transformer model
# (turns text into a 384-dimension vector to match 'all-MiniLM-L6-v2', a model that balances speed & accuracy)
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_vector_search(json_path):
    print("Loading your lyrics dataset...")
    # 'lines=True' if JSON is one object per line, otherwise omit it
    df = pd.read_json(json_path) 
    # ensure the column names match JSON keys exactly
    df = df.dropna(subset=['lyrics']).reset_index(drop=True)
    print(f"Generating embeddings for {len(df)} songs...")
    # vectorization to capture semantics
    embeddings = model.encode(df['lyrics'].tolist(), show_progress_bar=True)
    # VECTOR DATABASE: initializing faiss (facebook ai similarity search)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # measures euclidean distance between song vectors
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, "song_vibe_index.faiss")
    # save the specific keys from your JSON for the metadata
    df[['title', 'genre']].to_pickle("song_metadata.pkl")

build_vector_search('song_data_small.json')