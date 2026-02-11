
import json
import re

from sklearn.model_selection import train_test_split


def clean_lyrics(song_lyrics):
    """Clean song lyrics by lowercasing, removing brackets/punctuation, normalizing whitespace, and splitting into lines."""

    # lowercase:
    song_lyrics = song_lyrics.lower()

    # get rid of square brackets & anything within:
    # .* matches any number of any char, and ? ensures that the expression stops when
    # it finds ]. re.DOTALL ensure that the . will catch ANY char, including newline
    song_lyrics = re.sub(r"\[.*?\]", " ", song_lyrics, flags = re.DOTALL)

    # replace hyphens with spaces (not perfect in every case, but will address something
    # like "state-of-the-art" better than if it were to become "stateoftheart")
    song_lyrics = re.sub(r"-", " ", song_lyrics)

    # get rid of other punctuation (anything that is not a word char \w, space char \s,
    # or apostrophe since that is important in words like contractions
    song_lyrics = re.sub(r"[^\w\s']", "", song_lyrics)

    # account for possible different newline characters
    song_lyrics = song_lyrics.replace("\r\n", "\n")
    song_lyrics = song_lyrics.replace("\r", "\n")

    # if any lines are separated by more than one newline, collapse it down to one
    song_lyrics = re.sub(r"\n+", "\n", song_lyrics)

    # get rid of extra spaces, in case any words are separated by extra whitespace
    # also, use .strip() to get rid of newline characters if they appear at the very start or
    # very end of the set of lyrics (this does not touch internal newlines in the lyrics)
    song_lyrics = re.sub(r"[ ]+", " ", song_lyrics).strip()

    # turn the song lyrics into a list of strings, one string per line
    song_lyrics = song_lyrics.split("\n")

    return song_lyrics


def all_cap_word_count(tokens):
    """Count the number of all-uppercase words longer than one character in a token list."""
    return sum(1 for word in tokens if len(word) > 1 and word.isupper())


def extract_non_ngram_features(text):
    """Extract counts of structural tags, punctuation, newlines, and all-caps words from text."""

    # minimal cleaning to remove punctuation while preserving case (mostly so that
    # the all caps correctly catches a word like "ROCK!" detached from the punctuation)
    clean_text = re.sub(r"\[.*?\]", " ", text, flags=re.DOTALL)
    clean_text = re.sub(r"-", " ", clean_text)
    clean_text = re.sub(r"[^\w\s']", "", clean_text)
    clean_text = re.sub(r"[ ]+", " ", clean_text).strip()

    text_lower = text.lower()

    # pass the minimally cleaned text to the counter
    counts = {
        "intro_tag_count": text_lower.count("[intro"), # intentionally omitting the closing bracket
        "hook_tag_count": text_lower.count("[hook"),
        "chorus_tag_count": text_lower.count("[chorus"),
        "verse_tag_count": text_lower.count("[verse"),
        "exclamation": text.count("!"),
        "question": text.count("?"),
        "newline": text.count("\n"),
        "all_cap": all_cap_word_count(clean_text.split()),  # use the minimally cleaned text split
    }
    return counts


def load_and_process_json(input_path, output_path):
    """Load a JSON dataset of songs, clean lyrics, extract features, save processed JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        song_data = json.load(f)

    # overwrite the current lyrics with a cleaner pre-processed set of lyrics
    for song in song_data:
        song["features"] = extract_non_ngram_features(song["lyrics"])
        song["lyrics"] = clean_lyrics(song["lyrics"])

    print("total songs loaded:", len(song_data))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(song_data, f, ensure_ascii=False)

    print("processed data saved to:", output_path)
    return song_data


def create_train_dev_test_splits(songs, output_path, random_state=42):
    """Split songs into train/dev/test sets and save as JSON."""

    # 70% for training, 30% for the rest
    train_songs, intermediate = train_test_split(
        songs, test_size=0.3, random_state=random_state, stratify=[song["genre"] for song in songs])
    # split that remaining 30% to become 15% dev and 15% test
    dev_songs, test_songs = train_test_split(
        intermediate, test_size=0.5, random_state=random_state, stratify=[song["genre"] for song in intermediate])

    # save splits to use for all configurations
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump((train_songs, dev_songs, test_songs), f, ensure_ascii=False)

    print("train/dev/test sizes:", len(train_songs), len(dev_songs), len(test_songs))


if __name__ == "__main__":
    # relative paths, just applied to the smaller sample data file:
    processed_songs = load_and_process_json("data/data_sample.json", "data/processed_data_sample.json")
    create_train_dev_test_splits(processed_songs, "data/splits_sample.json")