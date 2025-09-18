"""
Data Augmentation Module for HER2+ Breast Cancer Analysis

This module provides comprehensive data augmentation functionality for H&E-stained
whole-slide image analysis, including tissue detection, color augmentation,
and medical imaging specific transforms.

Key Features:
- Otsu-based tissue detection for WSI analysis
- H&E stain variation simulation
- MONAI-based medical imaging transforms
- Configurable augmentation parameters

Authors:
    - Primary: T. Buathongtanakarn
    - AI Assistant: GitHub Copilot

Version: 2.1.0
Last Updated: September 17, 2025
"""

# Environment compatibility
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Standard imports
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional
import torch
import torchvision.transforms as transforms
from skimage import filters, morphology
from skimage.color import rgb2gray

# MONAI imports for medical imaging transforms
try:
    import monai
    from monai.transforms import (
        Compose, EnsureChannelFirstd, RandRotated, RandFlipd, 
        RandGaussianNoised, RandAdjustContrastd, ScaleIntensityRanged,
        EnsureTyped
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("MONAI not available. Install with: pip install monai")

# Native H&E stain normalization (no external dependencies)
# Removed staintools dependency due to build issues - using native implementation

def rgb_to_od(rgb_image):
    """Convert RGB image to optical density (OD) space"""
    rgb_image = np.array(rgb_image).astype(np.float64)
    # Avoid log(0) by adding small epsilon
    rgb_image = np.maximum(rgb_image, 1e-6)
    od = -np.log(rgb_image / 255.0)
    return od

def od_to_rgb(od_image):
    """Convert optical density (OD) image back to RGB"""
    rgb = 255 * np.exp(-od_image)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb

def get_he_stain_matrix():
    """Get standard H&E stain matrix (Ruifrok & Johnston method)"""
    # Standard H&E stain vectors in OD space
    he_matrix = np.array([
        [0.65, 0.70, 0.29],  # Hematoxylin
        [0.07, 0.99, 0.11]   # Eosin
    ])
    return he_matrix

def separate_he_stains(rgb_image):
    """Separate H&E stains using color deconvolution"""
    od = rgb_to_od(rgb_image)
    od_reshaped = od.reshape(-1, 3)
    
    # H&E stain matrix
    he_matrix = get_he_stain_matrix()
    
    # Separate stains using least squares
    try:
        stain_concentrations = np.linalg.lstsq(he_matrix.T, od_reshaped.T, rcond=None)[0]
        h_concentration = stain_concentrations[0].reshape(od.shape[:2])
        e_concentration = stain_concentrations[1].reshape(od.shape[:2])
        return h_concentration, e_concentration
    except:
        # Fallback: simple separation
        h_concentration = 0.5 * (od[:,:,0] + od[:,:,1])
        e_concentration = 0.5 * (od[:,:,1] + od[:,:,2])
        return h_concentration, e_concentration

def normalize_he_stain(rgb_image, target_concentrations=None):
    """Native H&E stain normalization without external dependencies"""
    if target_concentrations is None:
        # Standard target concentrations
        target_concentrations = {'h_mean': 0.8, 'h_std': 0.15, 'e_mean': 0.6, 'e_std': 0.12}
    
    try:
        # Separate H&E stains
        h_conc, e_conc = separate_he_stains(rgb_image)
        
        # Normalize concentrations
        h_mean, h_std = np.mean(h_conc), np.std(h_conc)
        e_mean, e_std = np.mean(e_conc), np.std(e_conc)
        
        if h_std > 0:
            h_conc_norm = (h_conc - h_mean) / h_std * target_concentrations['h_std'] + target_concentrations['h_mean']
        else:
            h_conc_norm = h_conc
            
        if e_std > 0:
            e_conc_norm = (e_conc - e_mean) / e_std * target_concentrations['e_std'] + target_concentrations['e_mean']
        else:
            e_conc_norm = e_conc
        
        # Reconstruct RGB
        he_matrix = get_he_stain_matrix()
        od_norm = np.zeros_like(rgb_to_od(rgb_image))
        
        for i in range(3):
            od_norm[:,:,i] = h_conc_norm * he_matrix[0,i] + e_conc_norm * he_matrix[1,i]
        
        rgb_norm = od_to_rgb(od_norm)
        return rgb_norm
        
    except Exception as e:
        print(f"H&E normalization failed: {e}, returning original image")
        return np.array(rgb_image)

class NativeStainNormalizer:
    """Native stain normalizer to replace staintools"""
    
    def __init__(self):
        self.target_stats = None
        self.fitted = False
    
    def fit(self, reference_image):
        """Fit normalizer to reference image"""
        try:
            if isinstance(reference_image, Image.Image):
                reference_image = np.array(reference_image)
            
            h_conc, e_conc = separate_he_stains(reference_image)
            self.target_stats = {
                'h_mean': np.mean(h_conc),
                'h_std': np.std(h_conc),
                'e_mean': np.mean(e_conc),
                'e_std': np.std(e_conc)
            }
            self.fitted = True
        except Exception as e:
            print(f"Stain normalizer fit failed: {e}")
            self.fitted = False
    
    def transform(self, image):
        """Apply stain normalization"""
        if not self.fitted:
            return np.array(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        return normalize_he_stain(image, self.target_stats)


class AugmentationConfig:
    """Configuration for augmentation parameters"""
    ELASTIC_DEFORM_PROB = 0.3
    STAIN_AUGMENT_PROB = 0.5
    ROTATION_DEGREES = 90
    COLOR_JITTER_PARAMS = {
        'brightness': 0.1,
        'contrast': 0.1, 
        'saturation': 0.1,
        'hue': 0.05
    }
    GAUSSIAN_NOISE_STD = 0.1
    CONTRAST_GAMMA_RANGE = (0.8, 1.2)


# Tissue Detection Functions
def get_tissue_mask_otsu(slide_region: Image.Image, level: int = 6) -> np.ndarray:
    """
    Generate tissue mask using Otsu thresholding
    
    Args:
        slide_region: PIL Image of slide region
        level: Pyramid level for processing (higher = lower resolution)
        
    Returns:
        Binary tissue mask as uint8 array (255 = tissue, 0 = background)
    """
    try:
        # Convert to thumbnail for faster processing
        thumbnail = slide_region.copy()
        thumbnail.thumbnail((512, 512), Image.LANCZOS)
        
        # Convert to grayscale
        gray = rgb2gray(np.array(thumbnail))
        
        # Apply Otsu thresholding
        threshold = filters.threshold_otsu(gray)
        tissue_mask = gray < threshold  # Tissue is darker than background
        
        # Clean up mask with morphological operations
        tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=100)
        tissue_mask = morphology.binary_closing(tissue_mask, morphology.disk(5))
        
        return tissue_mask.astype(np.uint8) * 255
    except Exception as e:
        print(f"Error in tissue mask generation: {e}")
        return np.ones((512, 512), dtype=np.uint8) * 255


def extract_tissue_patches(slide, num_patches: int, patch_size: int, use_otsu: bool = True) -> List[Tuple[int, int]]:
    """
    Extract patch coordinates from tissue regions
    
    Args:
        slide: OpenSlide object
        num_patches: Number of patches to extract
        patch_size: Size of each patch
        use_otsu: Whether to use Otsu thresholding for tissue detection
        
    Returns:
        List of (x, y) coordinates for patch extraction
    """
    try:
        slide_dims = slide.level_dimensions[0]
        patch_coords = []
        
        if use_otsu and slide_dims[0] > patch_size and slide_dims[1] > patch_size:
            # Sample tissue regions using Otsu
            level = min(6, slide.level_count - 1)  # Use lower resolution for tissue detection
            
            # Get thumbnail
            thumbnail = slide.get_thumbnail((1024, 1024))
            tissue_mask = get_tissue_mask_otsu(thumbnail, level)
            
            # Find tissue coordinates
            tissue_coords = np.where(tissue_mask > 0)
            if len(tissue_coords[0]) > 0:
                # Scale coordinates back to full resolution
                scale_x = slide_dims[0] / tissue_mask.shape[1]
                scale_y = slide_dims[1] / tissue_mask.shape[0]
                
                for _ in range(num_patches):
                    if len(tissue_coords[0]) == 0:
                        break
                    idx = np.random.randint(len(tissue_coords[0]))
                    y, x = tissue_coords[0][idx], tissue_coords[1][idx]
                    
                    # Scale to full resolution
                    full_x = int(x * scale_x)
                    full_y = int(y * scale_y)
                    
                    # Ensure patch fits
                    full_x = min(full_x, slide_dims[0] - patch_size)
                    full_y = min(full_y, slide_dims[1] - patch_size)
                    
                    patch_coords.append((full_x, full_y))
        
        # Fallback to random sampling if Otsu fails or is disabled
        while len(patch_coords) < num_patches:
            x = np.random.randint(0, max(1, slide_dims[0] - patch_size))
            y = np.random.randint(0, max(1, slide_dims[1] - patch_size))
            patch_coords.append((x, y))
            
        return patch_coords[:num_patches]
        
    except Exception as e:
        print(f"Error in patch extraction: {e}")
        # Fallback to random sampling
        patch_coords = []
        for _ in range(num_patches):
            x = np.random.randint(0, max(1, slide_dims[0] - patch_size))
            y = np.random.randint(0, max(1, slide_dims[1] - patch_size))
            patch_coords.append((x, y))
        return patch_coords


# Stain Augmentation Functions
def elastic_deformation(image: np.ndarray, alpha: float = 100, sigma: float = 10) -> np.ndarray:
    """
    Apply elastic deformation to an image
    
    Args:
        image: Input image as numpy array
        alpha: Intensity of deformation
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Elastically deformed image
    """
    try:
        if len(image.shape) != 3:
            return image
            
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(*shape).astype(np.float32)
        dy = np.random.randn(*shape).astype(np.float32)
        
        # Smooth the displacement fields
        if sigma > 0:
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Scale by alpha
        dx *= alpha / shape[0]  # Normalize by image size
        dy *= alpha / shape[1]
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        
        # Add displacement
        map_x = x + dx
        map_y = y + dy
        
        # Apply deformation with proper interpolation
        deformed = cv2.remap(image, map_x, map_y, 
                           interpolation=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REFLECT_101)
        
        return deformed
    except Exception as e:
        print(f"Error in elastic deformation: {e}")
        return image


def random_he_augmentation(image, alpha_range=(0.7, 1.3), beta_range=(0.7, 1.3)) -> np.ndarray:
    """
    Apply realistic H&E stain augmentation using color deconvolution and reconstruction
    
    Args:
        image: Input image as numpy array (RGB) or PIL Image
        alpha_range: Range for hematoxylin stain variation  
        beta_range: Range for eosin stain variation
        
    Returns:
        Augmented image as numpy array
    """
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Ensure RGB format
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
            
        # H&E stain matrix (standard values)
        he_matrix = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11]   # Eosin
        ])
        
        # Convert to optical density (OD)
        od = -np.log((image.astype(np.float64) + 1) / 256.0)
        
        # Perform color deconvolution
        h_stain = np.dot(od, he_matrix[0])
        e_stain = np.dot(od, he_matrix[1])
        
        # Apply random augmentation
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        
        h_stain *= alpha
        e_stain *= beta
        
        # Reconstruct image
        reconstructed_od = np.outer(h_stain, he_matrix[0]) + np.outer(e_stain, he_matrix[1])
        reconstructed_od = reconstructed_od.reshape(image.shape)
        
        # Convert back to RGB
        augmented = (256 * np.exp(-reconstructed_od) - 1).clip(0, 255).astype(np.uint8)
        
        return augmented
        
    except Exception as e:
        print(f"Error in H&E augmentation: {e}")
        # Fallback to simple LAB augmentation
        try:
            # Convert to LAB color space for better stain separation
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Random perturbation of A and B channels (simulating H&E variation)
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            
            a = np.clip(a * alpha, 0, 255).astype(np.uint8)
            b = np.clip(b * beta, 0, 255).astype(np.uint8)
            
            # Merge back and convert to RGB
            lab = cv2.merge([l, a, b])
            augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return augmented
        except Exception as fallback_error:
            print(f"Fallback H&E augmentation also failed: {fallback_error}")
            return image
        b = np.clip(b * beta, 0, 255).astype(np.uint8)
        
        # Merge back and convert to RGB
        lab = cv2.merge([l, a, b])
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return augmented
    except Exception as e:
        print(f"Error in H&E augmentation: {e}")
        return image


def apply_stain_normalization(patch: Image.Image, stain_normalizer) -> Image.Image:
    """
    Apply native H&E stain normalization to a patch
    
    Args:
        patch: Input patch as PIL Image
        stain_normalizer: Native stain normalizer object
        
    Returns:
        Normalized patch as PIL Image
    """
    if stain_normalizer:
        patch_np = np.array(patch)
        normalized = stain_normalizer.transform(patch_np)
        return Image.fromarray(normalized)
    return patch


# Transform Definitions
def get_classification_transforms(phase: str = 'train') -> transforms.Compose:
    """
    Get torchvision transforms for classification tasks
    
    Args:
        phase: 'train' or 'val' to determine augmentation level
        
    Returns:
        Composed transforms
    """
    config = AugmentationConfig()
    
    if phase == 'train':
        transform_list = [
            transforms.RandomRotation(config.ROTATION_DEGREES),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(**config.COLOR_JITTER_PARAMS),
        ]
        
        # Add elastic deformation transform
        class ElasticTransform:
            def __init__(self, prob=0.3, alpha=100, sigma=10):
                self.prob = prob
                self.alpha = alpha
                self.sigma = sigma
                
            def __call__(self, image):
                if np.random.random() < self.prob:
                    # Convert PIL to numpy, apply elastic, convert back
                    if isinstance(image, Image.Image):
                        img_np = np.array(image)
                        deformed = elastic_deformation(img_np, self.alpha, self.sigma)
                        return Image.fromarray(deformed)
                    else:
                        return elastic_deformation(image, self.alpha, self.sigma)
                return image
        
        # Add H&E augmentation transform  
        class HETransform:
            def __init__(self, prob=0.5, alpha_range=(0.7, 1.3), beta_range=(0.7, 1.3)):
                self.prob = prob
                self.alpha_range = alpha_range
                self.beta_range = beta_range
                
            def __call__(self, image):
                if np.random.random() < self.prob:
                    if isinstance(image, Image.Image):
                        img_np = np.array(image)
                        augmented = random_he_augmentation(img_np, self.alpha_range, self.beta_range)
                        return Image.fromarray(augmented)
                    else:
                        return random_he_augmentation(image, self.alpha_range, self.beta_range)
                return image
        
        transform_list.extend([
            ElasticTransform(prob=config.ELASTIC_DEFORM_PROB),
            HETransform(prob=config.STAIN_AUGMENT_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    else:  # validation
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_segmentation_transforms(phase: str = 'train', elastic_deform_prob: float = 0.3) -> Optional[Compose]:
    """
    Get MONAI transforms for segmentation tasks
    
    Args:
        phase: 'train' or 'val' to determine augmentation level
        elastic_deform_prob: Probability of applying elastic deformation
        
    Returns:
        MONAI Compose object or None if MONAI not available
    """
    if not MONAI_AVAILABLE:
        print("MONAI not available for segmentation transforms")
        return None
        
    config = AugmentationConfig()
    
    if phase == 'train':
        return Compose([
            # Convert numpy arrays to tensors
            EnsureTyped(keys=['image', 'mask'], data_type='tensor'),
            # Add channel dimension if needed
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            # Elastic deformation
            monai.transforms.RandElasticd(
                keys=['image', 'mask'],
                sigma_range=(5, 7),
                magnitude_range=(100, 200),
                prob=elastic_deform_prob,
                rotate_range=0.1,
                shear_range=0.1,
                translate_range=10,
                mode=['bilinear', 'nearest'],
                padding_mode='border'
            ),
            RandRotated(keys=['image', 'mask'], range_x=0.1, prob=0.5, mode=['bilinear', 'nearest']),
            RandFlipd(keys=['image', 'mask'], spatial_axis=0, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=1, prob=0.5),
            RandGaussianNoised(keys=['image'], mean=0.0, std=config.GAUSSIAN_NOISE_STD, prob=0.2),
            RandAdjustContrastd(keys=['image'], gamma=config.CONTRAST_GAMMA_RANGE, prob=0.5),
            # Normalize image to [0,1] range
            ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            # Final type conversion
            EnsureTyped(keys=['image', 'mask'], data_type='tensor', dtype=torch.float32)
        ])
    else:  # validation
        return Compose([
            EnsureTyped(keys=['image', 'mask'], data_type='tensor'),
            EnsureChannelFirstd(keys=['image'], channel_dim=-1),
            ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=['image', 'mask'], data_type='tensor', dtype=torch.float32)
        ])


# Convenience Functions
def create_stain_normalizer(reference_image_path: str = None):
    """
    Create and fit a stain normalizer
    
    Args:
        reference_image_path: Path to reference image for normalization
        
    Returns:
        Fitted NativeStainNormalizer object
    """
    try:
        normalizer = NativeStainNormalizer()
        if reference_image_path:
            reference = Image.open(reference_image_path)
            normalizer.fit(np.array(reference))
        return normalizer
    except Exception as e:
        print(f"Error creating native stain normalizer: {e}")
        return None


def apply_random_augmentations(image: np.ndarray, 
                              apply_he_augment: bool = True,
                              he_alpha_range: Tuple[float, float] = (0.7, 1.3),
                              he_beta_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
    """
    Apply random augmentations to an image
    
    Args:
        image: Input image as numpy array
        apply_he_augment: Whether to apply H&E augmentation
        he_alpha_range: Range for H&E alpha channel perturbation
        he_beta_range: Range for H&E beta channel perturbation
        
    Returns:
        Augmented image
    """
    augmented = image.copy()
    
    # Apply H&E stain augmentation
    if apply_he_augment and np.random.random() < AugmentationConfig.STAIN_AUGMENT_PROB:
        augmented = random_he_augmentation(augmented, he_alpha_range, he_beta_range)
    
    return augmented


# Export main functions and classes
__all__ = [
    'AugmentationConfig',
    'get_tissue_mask_otsu',
    'extract_tissue_patches', 
    'random_he_augmentation',
    'elastic_deformation',
    'apply_stain_normalization',
    'get_classification_transforms',
    'get_segmentation_transforms',
    'create_stain_normalizer',
    'apply_random_augmentations'
]
