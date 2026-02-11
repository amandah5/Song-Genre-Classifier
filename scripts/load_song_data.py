
import json
from collections import Counter

import pandas as pd

def create_json(
    data_csv: str,
    output_file: str,
    allowed_genres: set = {"pop", "rap", "rock", "rb", "country"},
    limit_per_genre: int = 5000,
    chunk_size: int = 100000):

    """Reads a large CSV of song lyrics, filters for allowed genres and English language,
    and writes a balanced sample JSON file with a limited number of songs per genre."""

    counts = Counter()
    first_item = True

    # write to the json
    with open (output_file, "w", encoding="utf-8") as output:

        output.write("[")
        for chunk in pd.read_csv(data_csv, chunksize = chunk_size):

            # keep only the English data (mask that will take datapoints that yield True)
            chunk = chunk[chunk.language == "en"]

            # keep desired genres (mask to filter just the 5 genres)
            chunk = chunk[chunk.tag.isin(allowed_genres)]

            # get rid of any lines with NaN in the necessary fields
            chunk = chunk.dropna(subset = ["title", "tag", "lyrics"])

            for _, row in chunk.iterrows():
                genre = row["tag"]
                if counts[genre] < limit_per_genre:
                    song = {"title": row["title"], "genre": genre, "lyrics": row["lyrics"]}
                    if not first_item:
                        output.write(",\n") # ensure comma separated, but no comma after the last item
                    output.write(json.dumps(song, ensure_ascii = False)) # allow any character
                    counts[genre] += 1
                    first_item = False

            # no need to keep going if all limits have been reached
            if all(counts[g] >= limit_per_genre for g in allowed_genres):
                break
        output.write("]")

    print("Final counts per genre: ", counts)