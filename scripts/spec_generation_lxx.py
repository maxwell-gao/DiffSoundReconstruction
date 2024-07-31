import argparse
import io
import logging
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image as HfImage, Value
import librosa
from PIL import Image as PILImage
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")


def mel_spectrogram_to_image(S_db, top_db):
    """Converts a Mel spectrogram to a PIL Image with specified size."""
    S_db = np.clip(S_db, a_min=None, a_max=top_db)
    S_db_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
    S_db_image = PILImage.fromarray(S_db_normalized.astype(np.uint8))
    
    # Flip the image vertically
    S_db_image = S_db_image.transpose(PILImage.FLIP_TOP_BOTTOM)
    
    return S_db_image

def generate_spectrogram_image(audio_file, n_fft, hop_length, sample_rate, n_mels, top_db):
    # 加载音频
    y, sr = librosa.load(audio_file, sr=sample_rate)
    
    # 生成Mel频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 生成图像
    mel_image = mel_spectrogram_to_image(S_db, top_db)
    
    return mel_image

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.input_dir)
        for file in files
        if re.search(r"\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    examples = []
    try:
        for audio_file in tqdm(audio_files):
            try:
                mel_image = generate_spectrogram_image(
                    audio_file,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    sample_rate=args.sample_rate,
                    n_mels=args.n_mels,
                    top_db=args.top_db,
                )
                
                with io.BytesIO() as output:
                    mel_image.save(output, format="PNG")
                    bytes_data = output.getvalue()
                
                examples.append(
                    {
                        "image": {"bytes": bytes_data},
                        "audio_file": audio_file,
                        "slice": 0,  # 如果有切片操作，可以修改此处
                    }
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid audio files were found.")
            return
        ds = Dataset.from_pandas(
            pd.DataFrame(examples),
            features=Features(
                {
                    "image": HfImage(),
                    "audio_file": Value(dtype="string"),
                    "slice": Value(dtype="int16"),
                }
            ),
        )
        dsd = DatasetDict({"train": ds})
        dsd.save_to_disk(os.path.join(args.output_dir))
        if args.push_to_hub:
            dsd.push_to_hub(args.push_to_hub)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--resolution",
        type=str,
        default="256,256",
        help="Resolution of the output images (width,height).",
    )
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=16380)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--top_db", type=int, default=80)
    parser.add_argument("--n_mels", type=int, default=256)  # 添加n_mels参数
    parser.add_argument("--n_iter", type=int, default=50)   # 添加n_iter参数
    args = parser.parse_args()

    # Handle the resolutions.
    try:
        args.resolution = tuple(int(x) for x in args.resolution.split(","))
        if len(args.resolution) != 2:
            raise ValueError
    except ValueError:
        raise ValueError("Resolution must be a tuple of two integers.")

    main(args)
