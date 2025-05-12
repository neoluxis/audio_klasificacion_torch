import math
import numpy as np
import wavio
import os
from scipy.io import wavfile
from glob import glob
import matplotlib.pyplot as plt
import argparse
import librosa


def process_mel(src):
    """Generate Mel spectrogram from audio file."""
    rate, wav = wavfile.read(src)
    wav = wav.astype(np.float32, order="F")
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    wav = wav / np.max(np.abs(wav))
    mel_spectrogram = librosa.feature.melspectrogram(y=wav, sr=rate, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    print(mel_spectrogram_db.shape)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=rate, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()
    
def process_fft(src):
    """Generate FFT from audio file."""
    rate, wav = wavfile.read(src)
    wav = wav.astype(np.float32, order="F")
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    wav = wav / np.max(np.abs(wav))
    
    n = len(wav)
    k = np.arange(n)
    T = n / rate
    frq = k / T
    frq = frq[:n // 2]
    
    Y = np.fft.fft(wav) / n
    Y = Y[:n // 2]
    
    plt.figure(figsize=(10, 4))
    plt.plot(frq, abs(Y), 'r')
    plt.title('FFT of Audio Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

def main(args):
    src = args.src
    method = args.method
    if method == "mel":
        process_mel(src)
    elif method == "fft":
        process_fft(src)
    else:
        raise ValueError("Invalid method. Choose 'fft' or 'mel'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View audio FFT")
    parser.add_argument(
        "--src",
        type=str,
        default="clean/Bearded Seal/Bearded Seal_12.wav",
        help="Path to the source audio file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mel",
        choices=["fft", "mel"],
        help="Method to use for audio processing",
    )

    args = parser.parse_args()
    main(args)
