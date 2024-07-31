import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from datasets import load_from_disk, load_dataset
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from PIL import Image
import librosa
import soundfile as sf

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
        help="Directory to save the output images",
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
        default=16000,
        help="Sampling rate for audio generation",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=1024,
        help="FFT window size",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=128,
        help="Number of samples between successive frames",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=32,
        help="Number of iterations for Griffin-Lim algorithm",
    )
    parser.add_argument(
        "--top_db",
        type=int,
        default=80,
        help="Max decibels for spectrogram normalization",
    )
    return parser.parse_args()

# def load_vae_model(vae_path):
#     # Load the diffusion pipeline and extract the VAE model
#     pipeline = DiffusionPipeline.from_pretrained(vae_path)
#     vae_model = pipeline.vqvae
#     return vae_model

def load_vae_model(vae_path):
    try:
        vae_model = AutoencoderKL.from_pretrained(vae_path)
    except EnvironmentError:
        raise ValueError(f"Could not load the VAE model from the specified path: {vae_path}")
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

def transforms(examples, augmentations):
    images = [augmentations(image) for image in examples["image"]]
    filenames = [os.path.splitext(os.path.basename(file))[0] for file in examples["audio_file"]]
    return {"input": images, "filename": filenames}

def save_image(image_tensor, filepath):
    image = ToPILImage()(image_tensor.squeeze(0))
    image.save(filepath)

def image_to_audio(image: Image.Image, sr, n_fft, hop_length, n_iter, top_db) -> np.ndarray:
    """Converts spectrogram to audio.

    Args:
        image (`PIL Image`): x_res x y_res grayscale image

    Returns:
        audio (`np.ndarray`): raw audio
    """
    bytedata = np.frombuffer(image.tobytes(), dtype="uint8").reshape((image.height, image.width))
    log_S = bytedata.astype("float") * top_db / 255 - top_db
    S = librosa.db_to_power(log_S)
    audio = librosa.feature.inverse.mel_to_audio(
        S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter
    )
    return audio

def evaluate_vae(vae_model, test_dataloader, device, output_dir, sr, n_fft, hop_length, n_iter, top_db):
    vae_model.to(device)
    vae_model.eval()
    
    for idx, batch in enumerate(test_dataloader):
        clean_images = batch["input"].to(device)
        filenames = batch["filename"]
        with torch.no_grad():
            latent_representations = vae_model.encode(clean_images).latent_dist.sample()
            reconstructed_output = vae_model.decode(latent_representations)
        
        # Extract the tensor from the DecoderOutput
        reconstructed_data = reconstructed_output.sample
        
        # Ensure the tensor is on the CPU
        clean_images = clean_images.cpu()
        reconstructed_data = reconstructed_data.cpu()

        # Convert tensor to PIL image
        to_pil = ToPILImage()
        
        for i in range(len(clean_images)):
            original_image = to_pil(clean_images[i])
            reconstructed_image = to_pil(reconstructed_data[i])
            filename = filenames[i]
            
            # Save individual images
            save_image(clean_images[i], output_dir / f"original_{filename}.png")
            save_image(reconstructed_data[i], output_dir / f"reconstructed_{filename}.png")

            # Save comparison plots
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(original_image, cmap="gray")
            ax[0].set_title("Original Spectrogram")
            ax[0].axis("off")

            ax[1].imshow(reconstructed_image, cmap="gray")
            ax[1].set_title("Reconstructed Spectrogram")
            ax[1].axis("off")

            plt.savefig(output_dir / f"comparison_{filename}.png")
            plt.close()
            
            # Convert spectrogram to audio
            original_audio = image_to_audio(original_image, sr, n_fft, hop_length, n_iter, top_db)
            reconstructed_audio = image_to_audio(reconstructed_image, sr, n_fft, hop_length, n_iter, top_db)
            
            # Save audio
            sf.write(output_dir / f"original_{filename}.wav", original_audio, sr)
            sf.write(output_dir / f"reconstructed_{filename}.wav", reconstructed_audio, sr)

def main():
    args = parse_args()
    
    # Load VAE model
    vae_model = load_vae_model(args.vae)
    
    # Load test dataset
    test_dataset = load_test_dataset(args.test_dataset_name, args.test_dataset_config_name)
    
    # Define image transformations
    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])
    
    # Set transform for the test dataset
    test_dataset.set_transform(lambda examples: transforms(examples, augmentations))
    
    # Create test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate VAE model
    evaluate_vae(vae_model, test_dataloader, device, output_dir, args.sr, args.n_fft, args.hop_length, args.n_iter, args.top_db)

if __name__ == "__main__":
    main()
