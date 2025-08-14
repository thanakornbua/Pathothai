"""
Image processing module for manipulating large images.
Includes channel operations, filters, and transformations.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union
from skimage import filters, exposure, morphology
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

class ImageProcessor:
    """
    Advanced image processing for large images with channel manipulation.
    """
    
    def __init__(self):
        self.history = []  # For undo functionality
        self.max_history = 20
        
    def adjust_brightness_contrast(self, image: np.ndarray, brightness: float = 0, contrast: float = 1.0) -> np.ndarray:
        """
        Adjust brightness and contrast of an image.
        
        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast multiplier (0.0 to 3.0)
            
        Returns:
            Adjusted image
        """
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32)
            
            # Apply contrast
            img_float = img_float * contrast
            
            # Apply brightness
            img_float = img_float + brightness
            
            # Clip values to valid range
            img_float = np.clip(img_float, 0, 255)
            
            return img_float.astype(image.dtype)
            
        except Exception as e:
            print(f"Error adjusting brightness/contrast: {e}")
            return image
    
    def adjust_gamma(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to an image.
        
        Args:
            image: Input image
            gamma: Gamma value (0.1 to 3.0)
            
        Returns:
            Gamma-corrected image
        """
        try:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            
            # Apply gamma correction
            return cv2.LUT(image, table)
            
        except Exception as e:
            print(f"Error applying gamma correction: {e}")
            return image
    
    def adjust_hsv(self, image: np.ndarray, hue_shift: float = 0, saturation_scale: float = 1.0, 
                   value_scale: float = 1.0) -> np.ndarray:
        """
        Adjust HSV values of an image.
        
        Args:
            image: Input RGB image
            hue_shift: Hue shift in degrees (-180 to 180)
            saturation_scale: Saturation multiplier (0.0 to 2.0)
            value_scale: Value multiplier (0.0 to 2.0)
            
        Returns:
            HSV-adjusted image
        """
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                return image
                
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust hue
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Adjust saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
            
            # Adjust value
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
            
            # Convert back to RGB
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return rgb
            
        except Exception as e:
            print(f"Error adjusting HSV: {e}")
            return image
    
    def apply_histogram_equalization(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input image
            method: 'global' or 'clahe' (Contrast Limited Adaptive Histogram Equalization)
            
        Returns:
            Equalized image
        """
        try:
            if len(image.shape) == 3:
                # For color images, apply to each channel
                result = np.zeros_like(image)
                for i in range(image.shape[2]):
                    if method == 'clahe':
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        result[:, :, i] = clahe.apply(image[:, :, i])
                    else:
                        result[:, :, i] = cv2.equalizeHist(image[:, :, i])
                return result
            else:
                # For grayscale images
                if method == 'clahe':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    return clahe.apply(image)
                else:
                    return cv2.equalizeHist(image)
                    
        except Exception as e:
            print(f"Error applying histogram equalization: {e}")
            return image
    
    def apply_filter(self, image: np.ndarray, filter_type: str, **kwargs) -> np.ndarray:
        """
        Apply various filters to the image.
        
        Args:
            image: Input image
            filter_type: Type of filter to apply
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered image
        """
        try:
            if filter_type == 'gaussian':
                sigma = kwargs.get('sigma', 1.0)
                return filters.gaussian(image, sigma=sigma, preserve_range=True).astype(image.dtype)
                
            elif filter_type == 'median':
                kernel_size = kwargs.get('kernel_size', 5)
                return cv2.medianBlur(image, kernel_size)
                
            elif filter_type == 'bilateral':
                d = kwargs.get('d', 9)
                sigma_color = kwargs.get('sigma_color', 75)
                sigma_space = kwargs.get('sigma_space', 75)
                return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
                
            elif filter_type == 'unsharp_mask':
                radius = kwargs.get('radius', 1.0)
                amount = kwargs.get('amount', 1.0)
                return filters.unsharp_mask(image, radius=radius, amount=amount, preserve_range=True).astype(image.dtype)
                
            elif filter_type == 'edge_enhance':
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                return cv2.filter2D(image, -1, kernel)
                
            elif filter_type == 'sobel':
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = np.sqrt(sobelx**2 + sobely**2)
                return np.clip(sobel, 0, 255).astype(image.dtype)
                
            else:
                print(f"Unknown filter type: {filter_type}")
                return image
                
        except Exception as e:
            print(f"Error applying filter {filter_type}: {e}")
            return image
    
    def extract_channel(self, image: np.ndarray, channel: Union[int, str]) -> np.ndarray:
        """
        Extract a specific channel from the image.
        
        Args:
            image: Input image
            channel: Channel index (int) or name ('red', 'green', 'blue', 'alpha')
            
        Returns:
            Single channel image
        """
        try:
            if len(image.shape) != 3:
                return image
                
            if isinstance(channel, str):
                channel_map = {'red': 0, 'green': 1, 'blue': 2, 'alpha': 3}
                channel = channel_map.get(channel.lower(), 0)
            
            if channel < image.shape[2]:
                return image[:, :, channel]
            else:
                return image[:, :, 0]  # Default to first channel
                
        except Exception as e:
            print(f"Error extracting channel: {e}")
            return image
    
    def combine_channels(self, channels: list, mode: str = 'rgb') -> np.ndarray:
        """
        Combine multiple single-channel images into a multi-channel image.
        
        Args:
            channels: List of single-channel images
            mode: Output mode ('rgb', 'rgba', 'hsv')
            
        Returns:
            Multi-channel image
        """
        try:
            if not channels:
                return None
                
            # Ensure all channels have the same shape
            height, width = channels[0].shape[:2]
            normalized_channels = []
            
            for channel in channels:
                if channel.shape[:2] != (height, width):
                    channel = cv2.resize(channel, (width, height))
                if len(channel.shape) == 3:
                    channel = channel[:, :, 0]  # Take first channel if multi-channel
                normalized_channels.append(channel)
            
            # Stack channels
            if mode == 'rgb' and len(normalized_channels) >= 3:
                result = np.stack(normalized_channels[:3], axis=2)
            elif mode == 'rgba' and len(normalized_channels) >= 4:
                result = np.stack(normalized_channels[:4], axis=2)
            else:
                # Default: stack all available channels
                result = np.stack(normalized_channels, axis=2)
            
            return result.astype(channels[0].dtype)
            
        except Exception as e:
            print(f"Error combining channels: {e}")
            return None
    
    def apply_colormap(self, image: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
        """
        Apply a colormap to a grayscale image.
        
        Args:
            image: Input grayscale image
            colormap: Matplotlib colormap name
            
        Returns:
            Colored image
        """
        try:
            if len(image.shape) == 3:
                # Convert to grayscale first
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Normalize to 0-1 range
            normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # Apply colormap
            cmap = plt.get_cmap(colormap)
            colored = cmap(normalized)
            
            # Convert to 8-bit RGB
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            return colored_rgb
            
        except Exception as e:
            print(f"Error applying colormap: {e}")
            return image
    
    def apply_threshold(self, image: np.ndarray, threshold_type: str = 'otsu', **kwargs) -> np.ndarray:
        """
        Apply various thresholding techniques.
        
        Args:
            image: Input image
            threshold_type: Type of thresholding
            **kwargs: Threshold-specific parameters
            
        Returns:
            Thresholded binary image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            if threshold_type == 'otsu':
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif threshold_type == 'adaptive_mean':
                block_size = kwargs.get('block_size', 11)
                c = kwargs.get('c', 2)
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
                
            elif threshold_type == 'adaptive_gaussian':
                block_size = kwargs.get('block_size', 11)
                c = kwargs.get('c', 2)
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
                
            elif threshold_type == 'manual':
                threshold_value = kwargs.get('threshold', 127)
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                
            else:
                print(f"Unknown threshold type: {threshold_type}")
                return gray
            
            return binary
            
        except Exception as e:
            print(f"Error applying threshold: {e}")
            return image
    
    def calculate_histogram(self, image: np.ndarray, bins: int = 256) -> dict:
        """
        Calculate histogram for each channel.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data
        """
        try:
            histograms = {}
            
            if len(image.shape) == 3:
                channels = ['Red', 'Green', 'Blue']
                if image.shape[2] == 4:
                    channels.append('Alpha')
                    
                for i, channel in enumerate(channels[:image.shape[2]]):
                    hist, bin_edges = np.histogram(image[:, :, i], bins=bins, range=(0, 255))
                    histograms[channel] = {'hist': hist, 'bins': bin_edges}
            else:
                hist, bin_edges = np.histogram(image, bins=bins, range=(0, 255))
                histograms['Grayscale'] = {'hist': hist, 'bins': bin_edges}
            
            return histograms
            
        except Exception as e:
            print(f"Error calculating histogram: {e}")
            return {}
    
    def resize_image(self, image: np.ndarray, new_size: Tuple[int, int], method: str = 'lanczos') -> np.ndarray:
        """
        Resize image with various interpolation methods.
        
        Args:
            image: Input image
            new_size: (width, height) of output image
            method: Interpolation method
            
        Returns:
            Resized image
        """
        try:
            method_map = {
                'nearest': cv2.INTER_NEAREST,
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'lanczos': cv2.INTER_LANCZOS4,
                'area': cv2.INTER_AREA
            }
            
            interpolation = method_map.get(method, cv2.INTER_LANCZOS4)
            return cv2.resize(image, new_size, interpolation=interpolation)
            
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new bounding box
            cos_val = abs(rotation_matrix[0, 0])
            sin_val = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
            return rotated
            
        except Exception as e:
            print(f"Error rotating image: {e}")
            return image
