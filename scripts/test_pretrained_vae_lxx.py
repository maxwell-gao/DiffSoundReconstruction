import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from datasets import load_from_disk, load_dataset
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VAE model with test dataset.")
    parser.add_argument(
        "--vae",
        type=str,
        required=True,
        help="Path or identifier to pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        required=True,
        help="Path to the test dataset on disk",
    )
    parser.add_argument(
        "--test_dataset_config_name",
        type=str,
        default=None,
        help="Configuration name for the test dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output images and audio",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16380,
        help="Sampling rate for audio generation",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="FFT window size",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="Number of samples between successive frames",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="Number of iterations for Griffin-Lim algorithm",
    )
    parser.add_argument(
        "--top_db",
        type=int,
        default=80,
        help="Max decibels for spectrogram normalization",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=256,
        help="Number of Mel bands",
    )
    return parser.parse_args()

def load_vae_model(vae_path):
    try:
        vae_model = AutoencoderKL.from_pretrained(vae_path)
    except:
        pipeline = DiffusionPipeline.from_pretrained(vae_path)
        vae_model = pipeline.vqvae
    return vae_model

def load_test_dataset(test_dataset_path, config_name=None):
    if os.path.exists(test_dataset_path):
        test_dataset = load_from_disk(test_dataset_path, storage_options=config_name)["train"]
    else:
        test_dataset = load_dataset(
            test_dataset_path,
            config_name,
            cache_dir=None,
            split="train",
        )
    return test_dataset

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

def evaluate_vae(args):
    # Load VAE model
    vae_model = load_vae_model(args.vae)
    
    # Load test dataset
    test_dataset = load_test_dataset(args.test_dataset_name, args.test_dataset_config_name)
    
    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    for example in test_dataset:
        # Get the image from the example
        image = example["image"]
        
        # Convert image to tensor
        image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension
        
        # Encode the image to latent representation
        latent_representations = vae_model.encode(image_tensor).latent_dist.sample()
        
        # Decode the latent representation to reconstruct the image
        reconstructed_output = vae_model.decode(latent_representations)
        
        # Extract the tensor from the DecoderOutput
        reconstructed_data = reconstructed_output.sample
        
        # Convert tensors back to PIL Images
        original_image = image
        reconstructed_image = ToPILImage()(reconstructed_data.squeeze(0))
        
        original_buffer = BytesIO()
        original_image.save(original_buffer, format="PNG")
        original_image_byte = original_buffer.getvalue()
        
        reconstructed_buffer = BytesIO()
        reconstructed_image.save(reconstructed_buffer, format="PNG")
        reconstructed_image_byte = reconstructed_buffer.getvalue()
        
        # Save individual images
        original_image.save(os.path.join(args.output_dir, f"original_{os.path.basename(example['audio_file']).replace('.wav', '')}.png"))
        reconstructed_image.save(os.path.join(args.output_dir, f"reconstructed_{os.path.basename(example['audio_file']).replace('.wav', '')}.png"))
        
        # Save comparison plots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_image, cmap="gray")
        ax[0].set_title("Original Spectrogram")
        ax[0].axis("off")
        ax[1].imshow(reconstructed_image, cmap="gray")
        ax[1].set_title("Reconstructed Spectrogram")
        ax[1].axis("off")
        plt.savefig(os.path.join(args.output_dir, f"comparison_{os.path.basename(example['audio_file']).replace('.wav', '')}.png"))
        plt.close()
        
        Original_mel = byte_to_mel(original_image_byte, args.top_db)
        # 使用 Griffin-Lim 算法从 Mel 频谱图还原音频
        Original_audio = mel_spectrogram_to_audio(Original_mel, args.sr, args.n_fft, args.hop_length, args.n_iter)
        
        Reconstructed_mel = byte_to_mel(reconstructed_image_byte, args.top_db)
        # 使用 Griffin-Lim 算法从 Mel 频谱图还原音频
        Reconstructed_audio = mel_spectrogram_to_audio(Reconstructed_mel, args.sr, args.n_fft, args.hop_length, args.n_iter)
                
        # Save original audio file
        output_audio_path_original = os.path.join(args.output_dir, f"original_{os.path.basename(example['audio_file']).replace('.wav', '')}.wav")
        sf.write(output_audio_path_original, Original_audio, args.sr)
        
        # Save reconstructed audio file
        output_audio_path_reconstructed = os.path.join(args.output_dir, f"reconstructed_{os.path.basename(example['audio_file']).replace('.wav', '')}.wav")
        sf.write(output_audio_path_reconstructed, Reconstructed_audio, args.sr)


if __name__ == "__main__":
    args = parse_args()
    evaluate_vae(args)
