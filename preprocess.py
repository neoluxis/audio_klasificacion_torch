import numpy as np
import librosa

def process_raw(audio, sr):
    """
    Return the raw audio waveform.
    
    Args:
        audio: NumPy array of audio samples (shape: [N] or [1, N], mono, normalized).
        sr: Sample rate (e.g., 16000 Hz).
    
    Returns:
        np.ndarray: Raw waveform (shape: [N], float32).
    """
    return audio.flatten().astype(np.float32)

def process_mel(audio, sr, n_mels=128):
    """
    Generate Mel spectrogram from audio array.
    
    Args:
        audio: NumPy array of audio samples (shape: [N] or [1, N], mono, normalized).
        sr: Sample rate (e.g., 16000 Hz).
        n_mels: Number of Mel bands (default: 128).
    
    Returns:
        np.ndarray: Mel spectrogram in dB (shape: [n_mels, time_frames]).
    """
    audio = audio.flatten().astype(np.float32)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

def process_fft(audio, sr):
    """
    Generate FFT magnitudes from audio array.
    
    Args:
        audio: NumPy array of audio samples (shape: [N] or [1, N], mono, normalized).
        sr: Sample rate (e.g., 16000 Hz).
    
    Returns:
        tuple: (frequencies, magnitudes)
            - frequencies: np.ndarray of frequency bins (shape: [N//2]).
            - magnitudes: np.ndarray of FFT magnitudes (shape: [N//2]).
    """
    audio = audio.flatten().astype(np.float32)
    n = len(audio)
    k = np.arange(n)
    T = n / sr
    frq = k / T
    frq = frq[:n // 2]
    
    Y = np.fft.fft(audio) / n
    Y = Y[:n // 2]
    
    return frq, np.abs(Y)