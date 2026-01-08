import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

class GTZANDataset:
    def __init__(self, audio_dir, augmented_dir, sample_rate=22050, n_mfcc=20):
        self.audio_dir = audio_dir
        self.augmented_dir = augmented_dir
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
        os.makedirs(self.augmented_dir, exist_ok=True)

    def augment_and_extract(self):
        X, y = [], []

        # Loop through genre folders
        for genre in os.listdir(self.audio_dir):
            genre_path = os.path.join(self.audio_dir, genre)
            if not os.path.isdir(genre_path):
                continue
            files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            
            for idx, f in enumerate(files):
                filepath = os.path.join(genre_path, f)
                y_label = genre  # label = folder name

                # Load audio
                try:
                    y_audio, sr = librosa.load(filepath, sr=self.sr)
                except Exception as e:
                    print(f"Skipping {filepath}: {e}")
                    continue

                # Augmentation: original + stretch + pitch + noise
                augmented = [
                    y_audio,
                    librosa.effects.time_stretch(y_audio, rate=1.1),
                    librosa.effects.pitch_shift(y_audio, n_steps=2, sr=sr),
                    y_audio + 0.005 * np.random.randn(len(y_audio))
                ]

                for i, y_aug in enumerate(augmented):
                    # Extract MFCCs
                    mfcc = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=self.n_mfcc)
                    mfcc = mfcc.T  # time x n_mfcc
                    # Standardize
                    scaler = StandardScaler()
                    mfcc = scaler.fit_transform(mfcc)
                    X.append(mfcc)
                    y.append(y_label)

                if idx % 50 == 0:
                    print(f"Processed {idx+1}/{len(files)} files in genre {genre}")

        X = np.array(X, dtype=object)
        y = np.array(y)
        return X, y
