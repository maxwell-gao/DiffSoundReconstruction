import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from datasets import load_from_disk, load_dataset
from diffusers import DiffusionPipeline
from PIL import Image
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

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
        default=22050,
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
        default=512,
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
    pipeline = DiffusionPipeline.from_pretrained(vae_path)
    vae_model = pipeline.vqvae
    return vae_model

def load_test_dataset(test_dataset_path, config_name=None):
    if os.path.exists(test_dataset_path):
        test_dataset = load_from_disk(test_dataset_path, storage_options=config_name)["test"]
    else:
        test_dataset = load_dataset(
            test_dataset_path,
            config_name,
            cache_dir=None,
            split="test",
        )
    return test_dataset

def mel_spectrogram_to_image(S_db, top_db):
    """Converts a Mel spectrogram to a PIL Image."""
    S_db = np.clip(S_db, a_min=None, a_max=top_db)
    S_db_normalized = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
    S_db_image = Image.fromarray(S_db_normalized.astype(np.uint8))
    return S_db_image

def image_to_mel_spectrogram(image: Image.Image, top_db) -> np.ndarray:
    """Converts a PIL Image back to a Mel spectrogram."""
    bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
    log_S = bytedata.astype("float") * top_db / 255 - top_db
    S = librosa.db_to_power(log_S)
    return S

def mel_spectrogram_to_audio(S, sr, n_fft, hop_length, n_iter) -> np.ndarray:
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
        
        # Print image resolutions
        # print(f"Original Image Resolution: {original_image.size}")
        # print(f"Reconstructed Image Resolution: {reconstructed_image.size}")
        
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
        
        # Convert spectrogram images to audio
        original_S_db = image_to_mel_spectrogram(original_image, args.top_db)
        reconstructed_S_db = image_to_mel_spectrogram(reconstructed_image, args.top_db)
        
        y, _ = librosa.load(example["audio_file"], sr=args.sr)
        # Convert original spectrogram to audio
        y_original = mel_spectrogram_to_audio(original_S_db, args.sr, args.n_fft, args.hop_length, args.n_iter)
        # Ensure original audio length matches original audio length
        y_original = librosa.util.fix_length(y_original, size=len(y))
        
        y_reconstructed = mel_spectrogram_to_audio(reconstructed_S_db, args.sr, args.n_fft, args.hop_length, args.n_iter)
        y_reconstructed = librosa.util.fix_length(y_reconstructed, size=len(y))
        
        # Save original audio file
        output_audio_path_original = os.path.join(args.output_dir, f"original_{os.path.basename(example['audio_file']).replace('.wav', '')}.wav")
        sf.write(output_audio_path_original, y_original, args.sr)
        
        # Save reconstructed audio file
        output_audio_path_reconstructed = os.path.join(args.output_dir, f"reconstructed_{os.path.basename(example['audio_file']).replace('.wav', '')}.wav")
        sf.write(output_audio_path_reconstructed, y_reconstructed, args.sr)


if __name__ == "__main__":
    args = parse_args()
    evaluate_vae(args)
