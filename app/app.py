import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# LOAD MODEL & INDEXED ASSETS (faiss, pkl)
@st.cache_resource 
def load_assets():
    # load transformer-based encoder (all-MiniLM-L6-v2)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # load faiss index (library for vector search to approximate nearest neighbor)
    index = faiss.read_index("semantic_model/song_vibe_index.faiss")
    # load metadata dataframe (titles & genres)
    metadata = pd.read_pickle("semantic_model/song_metadata.pkl")
    return model, index, metadata

model, index, metadata = load_assets()

# USER INTERFACE
st.title("🎵 Semantic Lyric Search 🎵")
st.markdown("Query the dataset using natural language to find songs with similar thematic or emotional intent.")

query = st.text_input("Enter a search query (e.g., 'existential dread in an urban setting' or 'triumphant return home'):")

if query:
    # SEMANTIC ENCODING & SEARCH
    # convert natural language input into a high-dimensional vector
    query_vector = model.encode([query])
    # execute L2 distance search in the vector space for the top 5 matches
    distances, indices = index.search(query_vector, k=5)
    # DISPLAY RESULTS
    st.subheader("Top Semantic Matches:")
    for i in indices[0]:
        song = metadata.iloc[i]
        st.write(f"**{song['title']}** — *Genre: {song['genre']}*") # markdown