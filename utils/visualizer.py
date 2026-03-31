import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_spectrum(image_tensor, save_path=None):
    """Expects image_tensor shape: [C, H, W] in range [0, 1]"""
    # Convert to grayscale for overall spectral analysis
    gray_img = image_tensor.mean(dim=0) 
    
    fft_complex = torch.fft.fft2(gray_img)
    fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
    magnitude = torch.log(torch.abs(fft_shifted) + 1e-8).numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap='viridis')
    plt.title("Log Magnitude Spectrum")
    plt.colorbar()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()