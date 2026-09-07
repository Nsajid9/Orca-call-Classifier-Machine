import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def create_dirs(base_path):
    folders = [
        "train/call",
        "train/no_call",
        "test/call",
        "test/no_call"
    ]
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def generate_spectrograms(wav_file, out_dir, num_samples=240):
    print(f"Loading {wav_file}...")
    y, sr = librosa.load(wav_file)
    
    # We will just split the audio into small random chunks
    chunk_duration_s = 2.0
    chunk_length = int(sr * chunk_duration_s)
    
    if len(y) < chunk_length:
        print("Audio too short")
        return
    
    print(f"Generating {num_samples} spectrograms...")
    for i in range(num_samples):
        # Pick a random start point
        start = random.randint(0, len(y) - chunk_length)
        chunk = y[start : start + chunk_length]
        
        # Determine label and split
        is_call = random.choice([True, False])
        is_train = random.random() < 0.8
        
        split = "train" if is_train else "test"
        label = "call" if is_call else "no_call"
        
        # Generate Mel Spectrogram
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        
        # Save as image without borders or axes
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_axis_off()
        librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
        
        filename = f"spect_{i}.png"
        filepath = os.path.join(out_dir, split, label, filename)
        
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        if (i+1) % 50 == 0:
            print(f"Generated {i+1}/{num_samples}")

if __name__ == "__main__":
    base_dir = "Spectrograms"
    create_dirs(base_dir)
    wav_path = "hmpback1.wav"
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found.")
    else:
        generate_spectrograms(wav_path, base_dir, num_samples=240)
        print("Dataset generation complete.")
