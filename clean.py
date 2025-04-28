import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample as librosa_resample, to_mono
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

import wavio

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 20), min_periods=1, center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    try:
        obj = wavio.read(path)
        wav = obj.data.astype(np.float32, order="F")
        rate = obj.rate
        try:
            channel = wav.shape[1]
            if channel == 2:
                wav = to_mono(wav.T)
            elif channel == 1:
                wav = to_mono(wav.reshape(-1))
        except IndexError:
            wav = to_mono(wav.reshape(-1))
        if rate != sr:
            wav = librosa_resample(wav, orig_sr=rate, target_sr=sr)
        wav = wav.astype(np.int16)
        return sr, wav
    except Exception as exc:
        raise Exception(f"Error processing {path}: {str(exc)}") from exc


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split(".wav")[0]
    dst_path = os.path.join(target_dir.split(".")[0], fn + "_{}.wav".format(str(ix)))
    if os.path.exists(dst_path):
        return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob("{}/**".format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if ".wav" in x]
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            try:
                rate, wav = downsample_mono(src_fn, args.sr)
                mask, y_mean = envelope(wav, rate, threshold=args.threshold)
                wav = wav[mask]
                delta_sample = int(dt * rate)

                if wav.shape[0] < delta_sample:
                    sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                    sample[: wav.shape[0]] = wav
                    save_sample(sample, rate, target_dir, fn, 0)
                else:
                    trunc = wav.shape[0] % delta_sample
                    for cnt, i in enumerate(
                        np.arange(0, wav.shape[0] - trunc, delta_sample)
                    ):
                        start = int(i)
                        stop = int(i + delta_sample)
                        sample = wav[start:stop]
                        save_sample(sample, rate, target_dir, fn, cnt)
            except Exception as exc:
                print(f"Error processing {src_fn}: {str(exc)}")


def test_threshold(args):
    src_root = args.src_root
    wav_paths = glob("{}/**".format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    if len(wav_path) != 1:
        print("audio file not found for sub-string: {}".format(args.fn))
        return
    rate, wav = downsample_mono(wav_path[0], args.sr)
    mask, env = envelope(wav, rate, threshold=args.threshold)
    plt.style.use("ggplot")
    plt.title("Signal Envelope, Threshold = {}".format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color="r", label="remove")
    plt.plot(wav[mask], color="c", label="keep")
    plt.plot(env, color="m", label="envelope")
    plt.grid(False)
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleaning audio data")
    parser.add_argument(
        "--src_root",
        type=str,
        default="audio2025_rec",
        help="directory of audio files in total duration",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default="clean",
        help="directory to put audio files split by delta_time",
    )
    parser.add_argument(
        "--delta_time",
        "-dt",
        type=float,
        default=1.0,
        help="time in seconds to sample audio",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="rate to downsample audio, default to 16000",
    )
    parser.add_argument(
        "--fn",
        type=str,
        default="3a3d0279",
        help="file to plot over time to check magnitude",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="threshold magnitude for np.int16 dtype",
    )
    args, _ = parser.parse_known_args()

    split_wavs(args)
