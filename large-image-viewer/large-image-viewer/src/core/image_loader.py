import numpy as np
from PIL import Image
import os

class ImageLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.image = None
        self.channels = None

    def load_image(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        try:
            self.image = Image.open(self.filepath)
            self.image = self.image.convert("RGBA")  # Ensure image is in RGBA format
            self.channels = self.split_channels(self.image)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {e}")

    def split_channels(self, image):
        """Split the image into its RGBA channels."""
        return np.array(image).transpose((2, 0, 1))  # Shape: (channels, height, width)

    def get_channel(self, index):
        """Get a specific channel (0: R, 1: G, 2: B, 3: A)."""
        if self.channels is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        return self.channels[index]

    def manipulate_channel(self, index, operation):
        """Apply a manipulation operation to a specific channel."""
        if self.channels is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        
        channel = self.get_channel(index)
        if operation == 'invert':
            self.channels[index] = 255 - channel
        elif operation == 'normalize':
            self.channels[index] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        else:
            raise ValueError("Unsupported operation. Use 'invert' or 'normalize'.")

    def save_image(self, output_path):
        """Reconstruct the image from channels and save it."""
        if self.channels is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        
        reconstructed_image = np.clip(self.channels.transpose((1, 2, 0)), 0, 255).astype(np.uint8)
        Image.fromarray(reconstructed_image).save(output_path)