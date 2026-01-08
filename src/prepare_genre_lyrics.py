import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
Y_PATH = os.path.join(DATA_DIR, "y_labels.npy")          
OUTPUT_PATH = os.path.join(DATA_DIR, "X_lyrics.npy")       # output: lyrics assigned to tracks

# LOAD GTZAN genrelabels
y_labels = np.load(Y_PATH)  # shape: (3996,)
print("Loaded labels, first 5:", y_labels[:5])

# DEFINE GENRE-RELATED LYRICS
genre_lyrics = {
    'blues': "Woke up this morning, feeling so low, my baby left me, now I'm all alone",
    'classical': "Softly the piano flows, a symphony of endless skies",
    'country': "Riding down the dusty road, guitar in hand, chasing dreams untold",
    'disco': "Dance all night under the sparkling lights, groove to the funky beats",
    'hiphop': "Flow so tight, rhymes ignite, spitting truth in the spotlight",
    'jazz': "Saxophone swings, midnight brings a smoky rhythm in the club",
    'metal': "Shredding riffs, thunder drums, screaming loud till kingdom comes",
    'pop': "Catchy tune in my head, dancing under neon lights tonight",
    'reggae': "Feel the sun, feel the sea, rhythm of freedom sets me free",
    'rock': "Guitar wails, drums collide, voices roar in the night"
}

# assign lyrics to each track
X_lyrics = np.array([genre_lyrics[genre] for genre in y_labels])

# TEST shape and types
assert X_lyrics.shape == y_labels.shape, "Lyrics array must match number of tracks"
assert all(isinstance(x, str) for x in X_lyrics), "All entries must be strings"
print("Lyrics assigned to tracks, shape =", X_lyrics.shape)

# SAVE
np.save(OUTPUT_PATH, X_lyrics)
print("Saved genre-related lyrics to:", OUTPUT_PATH)
