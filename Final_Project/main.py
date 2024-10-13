import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Step 1: Convert MPF to WAV
def mpf_to_wav(mpf_path, wav_path):
    # This function needs to be implemented based on the specific MPF format
    # For this example, we'll assume it's already done
    print(f"Converting {mpf_path} to {wav_path}")
    # audio = AudioSegment.from_file(mpf_path, format="mpf")
    # audio.export(wav_path, format="wav")

# Step 2: Load and preprocess the audio
def load_and_preprocess(audio_path):
    y, sr = librosa.load(audio_path)
    return y, sr

# Step 3: Extract features (STFT)
def extract_features(y, sr):
    S = np.abs(librosa.stft(y))
    return S

# Step 4: Apply NMF for source separation
def separate_sources(S, n_components=2):
    model = NMF(n_components=n_components, random_state=0)
    W = model.fit_transform(S.T)
    H = model.components_
    return W, H

# Step 5: Reconstruct and save separated sources
def reconstruct_and_save(W, H, sr, output_paths):
    for i, (w, h) in enumerate(zip(W.T, H)):
        source = librosa.istft(w.reshape(-1, 1) * h)
        sf.write(output_paths[i], source, sr)

# Step 6: Visualize results
def visualize_separation(original, separated):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(original, sr=sr)
    plt.title('Original Audio')
    
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(separated[0], sr=sr)
    plt.title('Separated Source 1 (Potential Human Voice)')
    
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(separated[1], sr=sr)
    plt.title('Separated Source 2 (Other Sounds)')
    
    plt.tight_layout()
    plt.show()

# Main process
def main():
    mpf_path = "path_to_your_file.mpf"
    wav_path = "converted_audio.wav"
    output_paths = ["separated_voice.wav", "separated_other.wav"]

    # Convert MPF to WAV
    mpf_to_wav(mpf_path, wav_path)

    # Load and preprocess
    y, sr = load_and_preprocess(wav_path)

    # Extract features
    S = extract_features(y, sr)

    # Separate sources
    W, H = separate_sources(S)

    # Reconstruct and save
    reconstruct_and_save(W, H, sr, output_paths)

    # Load separated sources for visualization
    separated = [librosa.load(path)[0] for path in output_paths]

    # Visualize
    visualize_separation(y, separated)

if __name__ == "__main__":
    main()