import argparse
import io
import logging
import os
import re
from io import BytesIO
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image as HfImage, Value
import librosa
from PIL import Image
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")


# 生成功率谱的Mel频谱图Array
def audio_to_mel(audio_path, sample_rate, n_fft, hop_length, n_mels):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

# 将 Mel 频谱图Array转换为PIL图像
def mel_spectrogram_to_image(S_db, top_db):
    """Converts a Mel spectrogram to a PIL Image."""
    S_db = np.clip(S_db, a_min=None, a_max=top_db)
    S_db_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
    S_db_image = Image.fromarray(S_db_normalized.astype(np.uint8))
    return S_db_image

# 从PIL图像提取Byte data
def image_to_byte(image: Image.Image,) -> np.ndarray:
    """Converts a PIL Image back to byte data."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    return byte_data

# 将Byte data还原回Mel 频谱图Array
def byte_to_mel(byte_data, top_db):
    """Converts byte data back to a Mel spectrogram."""
    image = Image.open(BytesIO(byte_data))
    bytedata = np.array(image).astype("float32")
    log_S = bytedata * top_db / 255 - top_db
    S = librosa.db_to_power(log_S)
    return S

# 从 Mel 频谱图Array恢复音频
def mel_spectrogram_to_audio(S, sr, n_fft, hop_length, n_iter):
    """Converts a Mel spectrogram to audio using Griffin-Lim."""
    audio = librosa.feature.inverse.mel_to_audio(
        S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter
    )
    return audio

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
                mel_array = audio_to_mel(
                    audio_file,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    sample_rate=args.sample_rate,
                    n_mels=args.n_mels,
                    
                )
                mel_PIL = mel_spectrogram_to_image(
                    mel_array,
                    top_db=args.top_db,
                )
                
                mel_byte = image_to_byte(mel_PIL)
                
                examples.extend(
                    [
                        {
                            "image": mel_byte,
                            "audio_file": audio_file,
                            "slice": 0,  # 如果有切片操作，可以修改此处
                        }
                    ]
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
    # parser.add_argument(
    #     "--resolution",
    #     type=str,
    #     default="256,256",
    #     help="Resolution of the output images (width,height).",
    # )
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--top_db", type=int, default=80)
    parser.add_argument("--n_mels", type=int, default=256)  # 添加n_mels参数
    parser.add_argument("--n_iter", type=int, default=50)   # 添加n_iter参数
    args = parser.parse_args()

    # Handle the resolutions.
    # try:
    #     args.resolution = tuple(int(x) for x in args.resolution.split(","))
    #     if len(args.resolution) != 2:
    #         raise ValueError
    # except ValueError:
    #     raise ValueError("Resolution must be a tuple of two integers.")

    main(args)
