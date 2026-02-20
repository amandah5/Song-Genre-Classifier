# Song Genre Classifier

This project is an exercise in data cleaning and feature engineering. 
Through a dataset containing millions of songs and their corresponding 
genres from the set {country, pop, rap, r&b, rock}, various models are
trained and compared in their classification success.

**New Feature: Semantic Lyric Search (Neural Indexing).** I have extended this project beyond simple classification to include a **Semantic Search Engine**. More on this below. 

The dataset for this project comes from Kaggle's [Genius Song Lyrics dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data), 
which builds on the [5 Million Song Lyrics](https://www.kaggle.com/datasets/nikhilnayak123/5-million-song-lyrics-dataset)
dataset. The lyrics were scraped from [genius.com](https://genius.com/).

***

### Genre classifier:

#### Implement by running these 3 scripts:
- python process_song_data.py 
- python feature_extraction.py 
- python run_models.py

**A note on load_song_data.py:** The get_sample.py script was used to generate 
the data_sample.json file from the full dataset. **The full dataset (~8 GB) is 
not included in this repo.**  
You do **not** need to run this script — all other scripts reference the included `data/data_sample.json` file.  

#### Script Overview:

- **load_song_data.py:** Loads the original dataset & cuts it down to include 
only 5000 instances in each of the 5 classes (25,000 songs total).

- **process_song_data.py:** Cleans each song's lyrics, splits it into 
lines & tokens, then creates train/dev/test split (70/15/15). This 
file also begins the feature extraction process for features that 
need to be recorded before/during cleaning, such as getting words 
in all caps before lowercasing everything, getting certain 
punctuation counts before stripping punctuation from tokens, etc.

- **feature_extraction.py:** Creates distinct feature sets based on binary 
unigram/bigram counts, tf-idf values, hand-crafted features, and a 
concatenation of the latter two, then passes these into DictVectorizer for all 
3 data splits.

- **run_models.py:** Performs the grid search with Naive Bayes and logistic 
regression, including the various feature sets and various hyperparameters 
(smoothing factor for NB, inverse L2 regularization strength for LR).<br>
<br>

***

### Semantic Search:
- **Vector Embeddings:** Uses the `all-MiniLM-L6-v2` Sentence-Transformer to map song lyrics into a 384-dimensional latent space.
- **Efficient Retrieval:** Utilizes a **FAISS** (Facebook AI Similarity Search) index to perform high-speed L2 distance searches across 25,000 song vectors.
- **Interactive UI:** A **Streamlit** dashboard allows users to query the dataset using natural language intent (e.g., "songs about late-night regret") rather than just keywords.

#### How to use the semantic search app:
- Generate the index by runnning `python app/sem_indexing.py`. This will encode the dataset and create the local vector store (approx. 15-30 mins on CPU).
- Run `streamlit run app/app.py` in your terminal to launch the app.
