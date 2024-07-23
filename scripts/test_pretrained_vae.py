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
from PIL import Image

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
    return parser.parse_args()

def load_vae_model(vae_path):
    # Load the diffusion pipeline and extract the VAE model
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

def transforms(examples, augmentations):
    images = [augmentations(image) for image in examples["image"]]
    return {"input": images}

def save_image(image_tensor, filepath):
    image = ToPILImage()(image_tensor.squeeze(0))
    image.save(filepath)

def evaluate_vae(vae_model, test_dataloader, device, output_dir):
    vae_model.to(device)
    vae_model.eval()
    
    for idx, batch in enumerate(test_dataloader):
        clean_images = batch["input"].to(device)
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
        
        # Save individual images
        save_image(clean_images[0], output_dir / f"original_{idx}.png")
        save_image(reconstructed_data[0], output_dir / f"reconstructed_{idx}.png")

        # Save comparison plots
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        ax[0].imshow(to_pil(clean_images[0]), cmap="gray")
        ax[0].set_title("Original Spectrogram")
        ax[0].axis("off")

        ax[1].imshow(to_pil(reconstructed_data[0]), cmap="gray")
        ax[1].set_title("Reconstructed Spectrogram")
        ax[1].axis("off")

        plt.savefig(output_dir / f"comparison_{idx}.png")
        plt.close()

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
    evaluate_vae(vae_model, test_dataloader, device, output_dir)

if __name__ == "__main__":
    main()
