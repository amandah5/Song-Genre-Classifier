
import json
import math
from collections import Counter

import numpy as np
from nltk.util import ngrams
from scipy.sparse import \
    hstack  # for feature matrix concatenation (last feature set)
from scipy.sparse import lil_matrix
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2


def padded_bigrams(tokens):
    """Return a list of bigrams from tokens with START and END tokens padded."""
    padded = ["<START>"] + tokens + ["<END>"]
    return ngrams(padded, 2) # nltk ngram function


def process_split(split_songs, top_unigrams, uni_idf, top_bigrams, bi_idf):
    """Process a list of songs to extract binary, TF-IDF, and hand-crafted feature sets."""
    bin_list = []
    tfidf_list = []
    mix_list = [] # hand-crafted features only
    for song in split_songs:
        b, t, m = extract_features(song, top_unigrams, uni_idf, top_bigrams, bi_idf)
        bin_list.append(b)
        tfidf_list.append(t)
        mix_list.append(m)
    return bin_list, tfidf_list, mix_list


def extract_features(song, top_unigrams, uni_idf, top_bigrams, bi_idf):
    """Extract binary, TF-IDF, and hand-crafted features from a song's tokens and lines."""

    features_bin = {}
    features_tfidf = {}
    features_mix = {}
    token_counts = Counter(song["all_tokens"])

    # get binary unigram/bigram counts & also find tf-idf values, using idf which was above
    for unigram in top_unigrams:
        count = token_counts.get(unigram, 0)
        if count > 0:
            uni_int = 1
            tf = 1 + math.log10(count) # smoothed
        else:
            uni_int = 0
            tf = 0
        if uni_int > 0: # saves some memory hopefully
            features_bin["UNIGRAM: " + str(unigram)] = uni_int
        features_tfidf["UNIGRAM: " + str(unigram)] = tf * uni_idf[unigram] # tf * idf

    bigram_counts = Counter()
    for line in song["tokenized_lines"]:
        bigram_counts.update([bigram for bigram in padded_bigrams(line) if bigram in top_bigrams])

    for bigram in top_bigrams:
        count = bigram_counts.get(bigram, 0)
        if count > 0:
            bi_int = 1
            tf = 1 + math.log10(count)
        else:
            bi_int = 0
            tf = 0
        if bi_int > 0:
            features_bin["BIGRAM: " + str(bigram[0]) + " " + str(bigram[1])] = bi_int
        features_tfidf["BIGRAM: " + str(bigram[0]) + " " + str(bigram[1])] = tf * bi_idf[bigram]

    # lastly: add the random chunk of features to the mix set (will be concatenated into the mega set also)
    for feature_name, val in song["features"].items():
        features_mix[feature_name] = val

    return features_bin, features_tfidf, features_mix



def run_feature_extraction(splits_file="data/splits_sample.json"):
    """Load song data, compute hand-crafted and n-gram features, vectorize, and return all feature sets and splits."""

    # load in the data (retrieve train/dev/test sets)
    with open(splits_file, "r", encoding="utf-8") as f:
        train_songs, dev_songs, test_songs = json.load(f)

    # all songs in one collection
    songs = train_songs + dev_songs + test_songs

    num_songs = len(train_songs) # just train! using this when deciding on n-grams to avoid leaking test data

    # hand-crafted features
    for song in songs:
        tokenized_lines = [line.split() for line in song["lyrics"] if line.strip()]
        song["tokenized_lines"] = tokenized_lines
        all_tokens = [token for line in tokenized_lines for token in line]
        song["all_tokens"] = all_tokens
        token_counts = Counter(all_tokens)
        features = song["features"]

        # reminder: song["features"] already contains some info from the previous file: punctuation, all caps, etc.

        # total number of lines in the song
        features["num_lines"] = len(tokenized_lines)

        # average line length, variance of line lengths (standard deviation)
        line_lengths = [len(line) for line in tokenized_lines]
        if line_lengths: # a.k.a. if this list is greater than 0
            avg = sum(line_lengths) / len(line_lengths)
            std = np.std(line_lengths)
        else:
            avg = 0
            std = 0
        features["average_line_length"] = avg
        features["line_length_std"] = std

        # unique word ratio, word length
        if len(all_tokens) != 0:
            features["unique_word_ratio"] = len(set(all_tokens)) / len(all_tokens)
            features["average_word_length"] = sum(len(word) for word in all_tokens) / len(all_tokens)
        else:
            features["unique_word_ratio"] = 0
            features["average_word_length"] = 0

        # repeating lines:
        line_counts = Counter((tuple(line) for line in tokenized_lines)) # error if I don't do this, need to make hashable ?
        repeated_line_count = sum(count - 1 for count in line_counts.values() if count > 1)
        # ^subtract 1 because if a line is occurring twice then technically it's only repeating once
        rep_line_proportion = (repeated_line_count / len(tokenized_lines) if tokenized_lines else 0)
        features["repeated_line_count"] = repeated_line_count
        features["repeated_line_proportion"] = rep_line_proportion

        # repeating words:
        word_counts = Counter(all_tokens)
        repeated_word_count = sum(count - 1 for count in word_counts.values() if count > 1)
        rep_word_proportion = (repeated_word_count / len(all_tokens) if all_tokens else 0)
        features["repeated_word_count"] = repeated_word_count
        features["repeated_word_proportion"] = rep_word_proportion

        # counts of genre-specific keywords
        keywords = {
            "rap_set": ["hood", "chain", "money", "bitch", "shit", "fuck", "gang", "homie", "dope", "yo", "hype"],
            "country_set": ["country", "truck", "road", "roads", "whiskey", "beer", "boots", "tractor", "dirt", "redneck", "jeans", "fields", "mama", "cowboy"],
            "pop_set": ["love", "heart", "feel", "tonight", "dj", "dance", "want", "perfect", "kiss"],
            "rock_set": ["rock", "fire", "break", "fight", "pain", "rage", "die", "fear", "dark", "cold"],
            "rb_set": ["baby", "touch", "close", "body", "sex", "sugar", "slow", "soul"],
        }
        for key, words in keywords.items():
            features[key] = sum(token_counts[word] for word in words)

    all_unigrams = Counter()
    all_bigrams = Counter()
    unigram_df = Counter()
    bigram_df = Counter()
    for song in train_songs:
        all_unigrams.update(song["all_tokens"])
        unigram_df.update(set(song["all_tokens"])) # for each document (song), just add 1 for every unigram
        song_bigrams = set()
        for line in song["tokenized_lines"]: # one line at a time for bigrams (not crossing lines)
            line_bigrams = list(padded_bigrams(line))
            all_bigrams.update(line_bigrams)
            song_bigrams.update(line_bigrams)
        bigram_df.update(song_bigrams)

    # before applying chi2: just take the top most 5000 of each (how many are there total? way more than this)
    candidate_unigrams = [word for word, count in all_unigrams.most_common(5000)]
    candidate_bigrams = [bigram for bigram, count in all_bigrams.most_common(5000)]

    # combine the unigrams & bigrams to make one larger n-gram vocab
    vocab = {ngram: index for index, ngram in enumerate(candidate_unigrams + candidate_bigrams)}

    # list of lists matrix ~ supposedly more efficient than np for my sparse binary/tfidf matrices
    X = lil_matrix((num_songs, len(vocab)), dtype = int) # 1 row per song, 1 column per n-gram in vocab
    all_labels = [] # will take label (genre) from every song

    for index, song in enumerate(train_songs):
        all_labels.append(song["genre"]) # keep track of the label for each data point

        # keep track of unigram counts (just have to count appearances in all_tokens)
        for token, count in Counter(song["all_tokens"]).items():
            if token in vocab: # if the token is in the pre-defined vocab above (just top 5000 of each n-gram)
                X[index, vocab[token]] = count # access the row (index) & the column (token), saved as the value in the vocab dict
                # ^assign the count of this token to the correct column in X

        # keep track of bigram counts (sum up counts from every line individually, since bigrams do not cross lines)
        for line in song["tokenized_lines"]:
            for bigram in padded_bigrams(line):
                if bigram in vocab: # same as unigram/token above^
                    X[index, vocab[bigram]] += 1
                    # increment the count for this bigram in the same row
                    # (adds together occurrences of the same bigram in different lines of the same song)

    label_map = {"rap": 0, "pop": 1, "rock": 2, "rb": 3, "country": 4}
    y_numeric = [label_map[genre] for genre in all_labels]

    X_csr = X.tocsr() # compressed sparse row (better for computations with sparse matrix)
    chi2_scores, _ = chi2(X_csr, y_numeric)

    # now take only the most informative n-grams out of the set of 10k (5k unigrams, 5k bigrams)
    # ***higher chi-squared values mean that a feature is more strongly indicative of a label***
    k = 4000
    # ^at k = 500, accuracy maxed at 59
    # ^at k = 1000, accuracy maxed at 61
    # ^at k = 2000, accuracy maxed around 63.9
    # ^at k = 3000, accuracy maxed at 65
    # ^at k = 5000, everything crashed and memory ran out.
    top_indices = np.argsort(chi2_scores)[::-1][:k] # reverse the list (to get highest scores first)

    # save a backwards dict of the (now shortened) vocab for easy value access
    index_to_ngram = {index: ngram for ngram, index in vocab.items()}
    top_ngrams = [index_to_ngram[ind] for ind in top_indices] # the most informative ones
    # ^this will be a mix of single token strings and tuples of 2 tokens, so separate them
    top_unigrams = [ngram for ngram in top_ngrams if isinstance(ngram, str)]
    top_bigrams = [ngram for ngram in top_ngrams if isinstance(ngram, tuple)]
    # (there will not necessarily be the same number of unigrams & bigrams since they were all merged before chi squared)

    # IDF values for top n-grams (including smoothing; this should match the TfidfVectorizer formula)
    uni_idf = {uni: math.log10((num_songs + 1) / (unigram_df[uni] + 1)) + 1 for uni in top_unigrams}
    bi_idf = {bi: math.log10((num_songs + 1) / (bigram_df[bi] + 1)) + 1 for bi in top_bigrams}

    splits_features = {
        "train": process_split(train_songs, top_unigrams, uni_idf, top_bigrams, bi_idf),
        "dev": process_split(dev_songs, top_unigrams, uni_idf, top_bigrams, bi_idf),
        "test": process_split(test_songs, top_unigrams, uni_idf, top_bigrams, bi_idf),
    }

    # already called extract features, just prepping lists of dicts for vectorizer
    bin_train = splits_features["train"][0]
    bin_dev = splits_features["dev"][0]
    bin_test = splits_features["test"][0]
    tfidf_train = splits_features["train"][1]
    tfidf_dev = splits_features["dev"][1]
    tfidf_test = splits_features["test"][1]
    mix_train = splits_features["train"][2]
    mix_dev = splits_features["dev"][2]
    mix_test = splits_features["test"][2]

    vectorizer_bin = DictVectorizer()
    vectorizer_tfidf = DictVectorizer()
    vectorizer_mix = DictVectorizer()

    features_bin_train = vectorizer_bin.fit_transform(bin_train)
    features_bin_dev = vectorizer_bin.transform(bin_dev)
    features_bin_test = vectorizer_bin.transform(bin_test)

    features_tfidf_train = vectorizer_tfidf.fit_transform(tfidf_train)
    features_tfidf_dev = vectorizer_tfidf.transform(tfidf_dev)
    features_tfidf_test = vectorizer_tfidf.transform(tfidf_test)

    features_mix_train = vectorizer_mix.fit_transform(mix_train)
    features_mix_dev = vectorizer_mix.transform(mix_dev)
    features_mix_test = vectorizer_mix.transform(mix_test)

    # scaling the mixed features bc they're all over the place (i got a warning message telling me to scale)
    scaler = preprocessing.StandardScaler()
    features_mix_train = scaler.fit_transform(features_mix_train.toarray()) # toarray makes them dense numpy arrays
    features_mix_dev = scaler.transform(features_mix_dev.toarray())         # (every entry needs a value, even if 0)
    features_mix_test = scaler.transform(features_mix_test.toarray())

    # concatenate tfidf & mix features horizontally (tfidf as is, mix after scaling):
    features_mega_train = hstack([features_tfidf_train, features_mix_train])
    features_mega_dev = hstack([features_tfidf_dev, features_mix_dev])
    features_mega_test = hstack([features_tfidf_test, features_mix_test])

    return {
        "features_bin_train": features_bin_train,
        "features_bin_dev": features_bin_dev,
        "features_bin_test": features_bin_test,
        "features_tfidf_train": features_tfidf_train,
        "features_tfidf_dev": features_tfidf_dev,
        "features_tfidf_test": features_tfidf_test,
        "features_mix_train": features_mix_train,
        "features_mix_dev": features_mix_dev,
        "features_mix_test": features_mix_test,
        "features_mega_train": features_mega_train,
        "features_mega_dev": features_mega_dev,
        "features_mega_test": features_mega_test,
        "train_songs": train_songs,
        "dev_songs": dev_songs,
        "test_songs": test_songs,
    }


if __name__ == "__main__":
    run_feature_extraction("data/splits_sample.json")
