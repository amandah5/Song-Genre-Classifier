import json
import random
from collections import defaultdict

input_file = "song_data_small.json"
output_file = "data_sample.json"
sample_per_class = 20  # number of songs per genre

# load full dataset
with open(input_file, "r", encoding="utf-8") as f:
    songs = json.load(f)

# group songs by genre
songs_by_genre = defaultdict(list)
for song in songs:
    genre = song.get("genre")
    songs_by_genre[genre].append(song)

# sample from each genre
sampled_songs = []
for genre, genre_songs in songs_by_genre.items():
    sampled_songs.extend(random.sample(genre_songs, min(sample_per_class, len(genre_songs))))

# save the smaller JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled_songs, f, ensure_ascii=False, indent=2)

print(f"Sampled {len(sampled_songs)} songs across {len(songs_by_genre)} genres and saved to {output_file}")