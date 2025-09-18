"""
HER2+ Breast Cancer Classification and Segmentation Training Pipeline

This module contains the core training functions for the HER2+ breast cancer 
classification and segmentation pipeline.

Key Features:
- Multi-phase training (ROI supervision → MIL → Segmentation)
- Advanced data augmentation
- Mixed precision training
- Comprehensive evaluation

Authors: T. Buathongtanakarn with assistance from GitHub Copilot (Claude 4.0 Model)

Version: 2.1.0
Last Updated: September 17, 2025
"""

# Environment setup for compatibility
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import collections.abc
if not hasattr(collections, 'Callable'):
    import collections
    collections.Callable = collections.abc.Callable

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
from torch.utils.checkpoint import checkpoint_sequential
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import openslide
import pydicom
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Progress bars
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# TensorBoard logging (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# MONAI for medical imaging
try:
    import monai
    from monai.data import DataLoader as MONAIDataLoader, Dataset as MONAIDataset
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
        ScaleIntensityRanged, RandRotated, RandFlipd, RandGaussianNoised,
        RandAdjustContrastd, RandGaussianSmoothd, RandCoarseDropoutd,
        ToTensord, EnsureTyped
    )
    from monai.networks.nets import UNet
    from monai.losses import DiceLoss, FocalLoss
    from monai.metrics import DiceMetric, ConfusionMatrixMetric
    from monai.utils import set_determinism
    from monai.inferers import sliding_window_inference
except ImportError:
    print("MONAI not installed. Please install with: pip install monai")
    sys.exit(1)

# W&B for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

# Explainability
try:
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    CAM_AVAILABLE = True
except ImportError:
    CAM_AVAILABLE = False
    print("PyTorch Grad-CAM not available. Install with: pip install pytorch-grad-cam")

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

# Helper function for conditional TensorBoard logging
def log_scalar(writer, tag: str, value, step: int):
    """Log scalar to TensorBoard if writer is available"""
    if writer is not None:
        writer.add_scalar(tag, value, step)

class MockWriter:
    """Mock TensorBoard writer for when TensorBoard is not available"""
    def add_scalar(self, *args, **kwargs):
        pass
    
    def close(self):
        pass

# Native H&E stain normalization (no external dependencies)
# Removed staintools dependency due to build issues

# Import augmentation functions
from scripts.augmentations import (
    get_tissue_mask_otsu,
    extract_tissue_patches,
    random_he_augmentation,
    elastic_deformation,
    apply_stain_normalization,
    get_classification_transforms,
    get_segmentation_transforms,
    create_stain_normalizer,
    apply_random_augmentations,
    AugmentationConfig
)

# Set deterministic behavior
set_determinism(seed=42)
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Data
    DATA_DIR = Path("data")
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    METADATA_FILE = DATA_DIR / "metadata.csv"
    PATCH_SIZE = 512
    PATCH_SIZE_SEG = 256  # Smaller for segmentation
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model
    NUM_CLASSES = 2  # HER2+ and HER2-
    BACKBONES = ['resnet50', 'efficientnet_b0']
    DROPOUT_RATE = 0.3
    ATTENTION_DIM = 128
    
    # Training
    EPOCHS_PHASE1 = 50
    EPOCHS_PHASE2 = 30
    LR_PHASE1 = 1e-4
    LR_PHASE2 = 1e-5
    LEARNING_RATE = 1e-4  # Alias for consistency
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5
    MIN_DELTA = 0.001
    MAX_EPOCHS = 50  # For consistency
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Logging
    LOG_DIR = Path("logs")
    CHECKPOINT_DIR = Path("checkpoints")
    WANDB_PROJECT = "her2-breast-cancer"
    TENSORBOARD_DIR = LOG_DIR / "tensorboard" if TENSORBOARD_AVAILABLE else None
    
    # Cross-validation
    N_FOLDS = 5
    
    # Optimization settings
    USE_TORCH_COMPILE = False  # Disable torch.compile to avoid Triton issues
    GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation for larger effective batch size
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    PREFETCH_FACTOR = 2  # DataLoader prefetch factor
    # Memory-savings (disabled by default to keep full resolution and trainable params)
    INPUT_SIZE = None  # When set (e.g., 224), tensors are resized before forward; None keeps original size
    LOW_MEM_MODE = False  # If True, freeze backbone and run it in eval mode to save memory
    GRADIENT_CHECKPOINT = True  # Enable gradient checkpointing to reduce memory usage
    
    # MIL and augmentation settings
    PATCHES_PER_SLIDE_PHASE1 = 100
    PATCHES_PER_SLIDE_PHASE2 = 200  # More patches for MIL
    PATCHES_PER_SLIDE_SEG = 50
    SLIDE_READ_LEVEL = 0  # Pyramid level to read patches from (0 = highest resolution)
    USE_OTSU_TISSUE_MASK = True
    ELASTIC_DEFORM_PROB = 0.3
    STAIN_AUGMENT_PROB = 0.5
    
    def __post_init__(self):
        """Create necessary directories"""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if TENSORBOARD_AVAILABLE and self.TENSORBOARD_DIR:
            self.TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items() 
                if not k.startswith('_')}

# Custom Dataset Classes
class HER2WSIDataset(Dataset):
    """Dataset for HER2 WSI patch extraction with ROI supervision"""
    
    def __init__(self, slide_paths: List[str], labels: List[int], 
                 annotations: List[Optional[str]], patch_size: int = 512,
                 patches_per_slide: int = 100,
                 slide_level: int = 0,
                 transform=None, stain_normalizer=None, phase: str = 'roi'):
        self.slide_paths = slide_paths
        self.labels = labels
        self.annotations = annotations
        self.patch_size = patch_size
        self.patches_per_slide = patches_per_slide
        self.slide_level = slide_level
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        self.phase = phase  # 'roi' or 'mil'
        
        # Cache for slides
        self.slide_cache = {}
        
        # Load annotations
        self.roi_coords = []
        for ann_path in annotations:
            if ann_path and os.path.exists(ann_path):
                coords = self._load_roi_annotations(ann_path)
                self.roi_coords.append(coords)
            else:
                self.roi_coords.append([])
    
    def _load_roi_annotations(self, xml_path: str) -> List[Tuple[int, int, int, int]]:
        """Load ROI coordinates from XML annotation file"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            coords = []
            
            for region in root.findall('.//Region'):
                vertices = []
                for vertex in region.findall('.//Vertex'):
                    x = int(float(vertex.get('X', 0)))
                    y = int(float(vertex.get('Y', 0)))
                    vertices.append((x, y))
                
                if len(vertices) >= 3:
                    # Get bounding box
                    xs = [v[0] for v in vertices]
                    ys = [v[1] for v in vertices]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    coords.append((min_x, min_y, max_x, max_y))
            
            return coords
        except Exception as e:
            print(f"Error loading annotations from {xml_path}: {e}")
            return []
    
    def _get_slide(self, slide_path: str):
        """Get cached slide object"""
        if slide_path not in self.slide_cache:
            self.slide_cache[slide_path] = openslide.OpenSlide(slide_path)
        return self.slide_cache[slide_path]
    
    def _extract_patch(self, slide, x: int, y: int, level: int = 0) -> Image.Image:
        """Extract patch from slide"""
        patch = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
        return patch.convert('RGB')
    
    def _apply_stain_normalization(self, patch: Image.Image) -> Image.Image:
        """Apply Macenko stain normalization or random H&E augmentation"""
        patch_np = np.array(patch)
        
        # Apply random H&E augmentation during training
        if self.phase == 'roi' and np.random.random() < Config.STAIN_AUGMENT_PROB:
            patch_np = random_he_augmentation(patch_np)
        
        # Apply fixed stain normalization if available
        if self.stain_normalizer:
            patch_np = self.stain_normalizer.transform(patch_np)
        
        return Image.fromarray(patch_np)
    
    def __len__(self):
        return len(self.slide_paths) * self.patches_per_slide
    
    def __getitem__(self, idx):
        slide_idx = idx // self.patches_per_slide
        patch_idx = idx % self.patches_per_slide
        
        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]
        roi_coords = self.roi_coords[slide_idx]
        
        slide = self._get_slide(slide_path)
        level = self.slide_level
        # Coordinates are in base-level reference frame for OpenSlide
        base_w, base_h = slide.level_dimensions[0]
        downsample = slide.level_downsamples[level] if hasattr(slide, 'level_downsamples') else (2 ** level)
        eff_size = int(self.patch_size * float(downsample))
        
        # Use tissue detection for better patch sampling in MIL phase
        if self.phase == 'mil' and Config.USE_OTSU_TISSUE_MASK:
            tissue_coords = extract_tissue_patches(slide, level, self.patch_size, use_otsu=True)
            if tissue_coords:
                x, y = tissue_coords[0]
            else:
                x = np.random.randint(0, max(0, base_w - eff_size))
                y = np.random.randint(0, max(0, base_h - eff_size))
        elif self.phase == 'roi' and roi_coords:
            # Extract from ROI
            roi = roi_coords[np.random.randint(len(roi_coords))]
            min_x, min_y, max_x, max_y = roi
            
            # Compute valid ranges ensuring patch fits within ROI bounds
            max_valid_x = max(min_x, max_x - eff_size)
            max_valid_y = max(min_y, max_y - eff_size)
            
            if max_valid_x >= min_x and max_valid_y >= min_y:
                # Allow exact-fit case and clamp to slide bounds
                x = np.random.randint(min_x, max_valid_x + 1) if max_valid_x > min_x else min_x
                y = np.random.randint(min_y, max_valid_y + 1) if max_valid_y > min_y else min_y
                # Clamp to slide dimensions just in case
                x = int(np.clip(x, 0, max(0, base_w - eff_size)))
                y = int(np.clip(y, 0, max(0, base_h - eff_size)))
            else:
                # Fallback to random if ROI is smaller than patch size
                x = np.random.randint(0, max(0, base_w - eff_size))
                y = np.random.randint(0, max(0, base_h - eff_size))
        else:
            # Random patch or tissue region
            x = np.random.randint(0, max(0, base_w - eff_size))
            y = np.random.randint(0, max(0, base_h - eff_size))
        
        # Extract and normalize patch
        patch = self._extract_patch(slide, x, y, level)
        patch = self._apply_stain_normalization(patch)
        
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label
    
    def __del__(self):
        """Clean up slide cache"""
        try:
            for slide in self.slide_cache.values():
                if hasattr(slide, 'close'):
                    slide.close()
            self.slide_cache.clear()
        except Exception:
            pass

class HER2SegmentationDataset(Dataset):
    """Dataset for segmentation with mask generation"""
    
    def __init__(self, slide_paths: List[str], annotations: List[Optional[str]], 
                 patch_size: int = 256, patches_per_slide: int = 50, slide_level: int = 0,
                 transform=None, stain_normalizer=None):
        self.slide_paths = slide_paths
        self.annotations = annotations
        self.patch_size = patch_size
        self.patches_per_slide = patches_per_slide
        self.slide_level = slide_level
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        
        self.slide_cache = {}
        self.mask_cache = {}
    
    def _get_slide_and_mask(self, slide_path: str, ann_path: Optional[str]):
        """Get slide and generate mask"""
        if slide_path not in self.slide_cache:
            self.slide_cache[slide_path] = openslide.OpenSlide(slide_path)
        
        slide = self.slide_cache[slide_path]
        
        if ann_path and ann_path not in self.mask_cache:
            # Generate mask from annotations
            mask = self._generate_mask_from_annotations(slide, ann_path)
            self.mask_cache[ann_path] = mask
        
        mask = self.mask_cache.get(ann_path, Image.new('L', slide.level_dimensions[0], 0))
        
        return slide, mask
    
    def _generate_mask_from_annotations(self, slide, xml_path: str) -> Image.Image:
        """Generate binary mask from XML annotations"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            slide_dims = slide.level_dimensions[0]
            mask = Image.new('L', slide_dims, 0)
            draw = ImageDraw.Draw(mask)
            
            for region in root.findall('.//Region'):
                vertices = []
                for vertex in region.findall('.//Vertex'):
                    x = int(float(vertex.get('X', 0)))
                    y = int(float(vertex.get('Y', 0)))
                    vertices.append((x, y))
                
                if len(vertices) >= 3:
                    draw.polygon(vertices, fill=255)  # Use 255 for consistency
            
            return mask
        except Exception as e:
            print(f"Error generating mask from {xml_path}: {e}")
            return Image.new('L', slide.level_dimensions[0], 0)
    
    def __len__(self):
        return len(self.slide_paths) * self.patches_per_slide
    
    def __getitem__(self, idx):
        slide_idx = idx // self.patches_per_slide
        patch_idx = idx % self.patches_per_slide
        
        slide_path = self.slide_paths[slide_idx]
        ann_path = self.annotations[slide_idx]
        
        slide, mask = self._get_slide_and_mask(slide_path, ann_path)
        
        # Random patch location in base coordinates for read_region
        base_w, base_h = slide.level_dimensions[0]
        downsample = slide.level_downsamples[self.slide_level] if hasattr(slide, 'level_downsamples') else (2 ** self.slide_level)
        eff_size = int(self.patch_size * float(downsample))
        
        x = np.random.randint(0, max(0, base_w - eff_size))
        y = np.random.randint(0, max(0, base_h - eff_size))
        
        # Extract patch and corresponding mask
        patch = slide.read_region((x, y), self.slide_level, (self.patch_size, self.patch_size)).convert('RGB')
        # Mask is defined at base resolution; crop uses base coords already
        mask_patch = mask.crop((x, y, x + eff_size, y + eff_size)).resize((self.patch_size, self.patch_size), Image.NEAREST)
        
        # Stain normalization/augmentation
        patch_np = np.array(patch)
        if np.random.random() < Config.STAIN_AUGMENT_PROB:
            patch_np = random_he_augmentation(patch_np)
        
        if self.stain_normalizer:
            patch_np = self.stain_normalizer.transform(patch_np)
        
        # Convert to format expected by MONAI transforms
        mask_np = np.array(mask_patch, dtype=np.uint8)
        mask_np = (mask_np > 127).astype(np.uint8)  # Ensure binary 0/1
        
        # Create data dict for MONAI transforms
        data_dict = {
            'image': patch_np,
            'mask': mask_np
        }
        
        if self.transform:
            try:
                data_dict = self.transform(data_dict)
                patch_tensor = data_dict['image']
                mask_tensor = data_dict['mask']
            except Exception as e:
                print(f"Transform error: {e}, falling back to simple conversion")
                # Fallback to simple tensor conversion
                patch_tensor = torch.from_numpy(patch_np.transpose(2, 0, 1)).float() / 255.0
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            # Simple tensor conversion
            patch_tensor = torch.from_numpy(patch_np.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        
        return patch_tensor, mask_tensor
    
    def __del__(self):
        """Clean up slide and mask cache"""
        try:
            for slide in self.slide_cache.values():
                if hasattr(slide, 'close'):
                    slide.close()
            self.slide_cache.clear()
            self.mask_cache.clear()
        except Exception:
            pass

# Model Architectures
class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning for slide-level classification"""
    
    def __init__(self, backbone: str = 'resnet50', num_classes: int = 2, 
                 attention_dim: int = 128, dropout: float = 0.3, use_checkpoint: bool = False):
        super().__init__()
        
        # Backbone
        if backbone == 'resnet50':
            self.feature_extractor = resnet50(pretrained=True)
            self.feature_extractor.fc = nn.Identity()
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.feature_extractor = efficientnet_b0(pretrained=True)
            self.feature_extractor.classifier = nn.Identity()
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store feature dimension and checkpointing flag
        self.feature_dim = feature_dim
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x):
        batch_size, num_instances, c, h, w = x.shape
        
        # Flatten instances
        x = x.view(batch_size * num_instances, c, h, w)
        
        # Extract features
        if self.use_checkpoint and self.training:
            # Break backbone into sequential blocks for checkpointing
            if isinstance(self.feature_extractor, nn.Sequential):
                modules = list(self.feature_extractor.children())
            else:
                # ResNet is nn.Module with layers; build a sequential
                fe = self.feature_extractor
                modules = [fe.conv1, fe.bn1, fe.relu, fe.maxpool,
                           fe.layer1, fe.layer2, fe.layer3, fe.layer4, fe.avgpool]
                self._feature_seq = nn.Sequential(*modules)
                modules = list(self._feature_seq.children())
            chunks = 4  # number of segments to checkpoint
            features = checkpoint_sequential(modules, chunks, x)
            features = torch.flatten(features, 1)
        else:
            features = self.feature_extractor(x)  # [batch_size * num_instances, feature_dim]
        features = features.view(batch_size, num_instances, -1)  # [batch_size, num_instances, feature_dim]
        
        # Attention weights
        attention_scores = self.attention(features)  # [batch_size, num_instances, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        attended_features = torch.sum(attention_weights * features, dim=1)  # [batch_size, feature_dim]
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits, attention_weights

class SegmentationUNet(nn.Module):
    """U-Net for tumor segmentation"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    
    def forward(self, x):
        return self.unet(x)

# Training Functions
def get_data_loaders(config: Config, fold: int = 0, phase: str = 'phase1'):
    """Create data loaders for training and validation"""
    
    # Load metadata
    if not config.METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.METADATA_FILE}")
    
    metadata = pd.read_csv(config.METADATA_FILE)
    
    # Get slide paths and labels
    slide_paths = []
    labels = []
    annotations = []
    
    ann_found_count = 0
    for _, row in metadata.iterrows():
        slide_name = row['slide_name']
        # Use slide_path from metadata if available, otherwise construct it
        if 'slide_path' in row and pd.notna(row['slide_path']):
            slide_path = Path(str(row['slide_path']))
        else:
            slide_path = config.DATA_DIR / f"{slide_name}.svs"
        
        if slide_path.exists():
            slide_paths.append(str(slide_path))
            labels.append(int(row['her2_status']))  # 0: HER2-, 1: HER2+
            
            # Resolve annotation file path preference: metadata column > config directory
            ann_path_obj = None
            try:
                if 'annotation_path' in metadata.columns:
                    ann_col_val = row.get('annotation_path', None)
                    if pd.notna(ann_col_val):
                        candidate = Path(str(ann_col_val))
                        if candidate.exists():
                            ann_path_obj = candidate
                if ann_path_obj is None:
                    candidate2 = config.ANNOTATIONS_DIR / f"{slide_name}.xml"
                    if candidate2.exists():
                        ann_path_obj = candidate2
            except Exception:
                # Be robust to any unexpected types in metadata
                ann_path_obj = None
            
            if ann_path_obj is not None and Path(ann_path_obj).exists():
                annotations.append(str(ann_path_obj))
                ann_found_count += 1
            else:
                annotations.append(None)

    # Quick visibility into ROI availability
    try:
        total_slides_ct = len(labels)
        print(f"[Data] Slides with ROI annotations: {ann_found_count}/{total_slides_ct}")
    except Exception:
        pass
    
    # Stratified split
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(slide_paths, labels))
    
    train_indices, val_indices = splits[fold]
    
    train_paths = [slide_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_annotations = [annotations[i] for i in train_indices]
    
    val_paths = [slide_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_annotations = [annotations[i] for i in val_indices]
    
    # Stain normalizer using imported function
    stain_normalizer = create_stain_normalizer()
    if stain_normalizer and config.DATA_DIR:
        # Fit normalizer on a reference image
        reference_path = config.DATA_DIR / "reference.png"
        if reference_path.exists():
            reference = Image.open(reference_path)
            stain_normalizer.fit(np.array(reference))
    
    # Transforms using imported functions
    if phase == 'phase1':
        # ROI-supervised transforms
        train_transform = get_classification_transforms('train')
        val_transform = get_classification_transforms('val')
        
        train_dataset = HER2WSIDataset(
            train_paths, train_labels, train_annotations,
            patch_size=config.PATCH_SIZE,
            patches_per_slide=config.PATCHES_PER_SLIDE_PHASE1,
            slide_level=config.SLIDE_READ_LEVEL,
            transform=train_transform,
            stain_normalizer=stain_normalizer, phase='roi'
        )
        
        val_dataset = HER2WSIDataset(
            val_paths, val_labels, val_annotations,
            patch_size=config.PATCH_SIZE,
            patches_per_slide=config.PATCHES_PER_SLIDE_PHASE1,
            slide_level=config.SLIDE_READ_LEVEL,
            transform=val_transform,
            stain_normalizer=stain_normalizer, phase='roi'
        )
        
    elif phase == 'segmentation':
        # Segmentation transforms using imported functions
        train_transform = get_segmentation_transforms('train', config.ELASTIC_DEFORM_PROB)
        val_transform = get_segmentation_transforms('val', config.ELASTIC_DEFORM_PROB)
        
        train_dataset = HER2SegmentationDataset(
            train_paths, train_annotations,
            patch_size=config.PATCH_SIZE_SEG,
            patches_per_slide=config.PATCHES_PER_SLIDE_SEG,
            slide_level=config.SLIDE_READ_LEVEL,
            transform=train_transform, stain_normalizer=stain_normalizer
        )
        
        val_dataset = HER2SegmentationDataset(
            val_paths, val_annotations,
            patch_size=config.PATCH_SIZE_SEG,
            patches_per_slide=config.PATCHES_PER_SLIDE_SEG,
            slide_level=config.SLIDE_READ_LEVEL,
            transform=val_transform, stain_normalizer=stain_normalizer
        )
    
    # Data loaders with optimizations
    train_loader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': config.NUM_WORKERS > 0
    }
    
    val_loader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': config.NUM_WORKERS > 0
    }
    
    # Add prefetch_factor only if num_workers > 0
    if config.NUM_WORKERS > 0:
        train_loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR
        val_loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    return train_loader, val_loader

def train_phase1(config: Config, fold: int = 0, writer = None):
    """Phase 1: ROI-supervised classification training"""
    
    # Initialize TensorBoard writer if available and not provided
    if writer is None and TENSORBOARD_AVAILABLE and config.TENSORBOARD_DIR:
        writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / f"phase1_fold{fold}"))
    
    # Model
    model = AttentionMIL(backbone='resnet50', num_classes=config.NUM_CLASSES,
                        attention_dim=config.ATTENTION_DIM, dropout=config.DROPOUT_RATE,
                        use_checkpoint=getattr(config, 'GRADIENT_CHECKPOINT', False))
    model = model.to(config.DEVICE)

    # Low-memory mode (optional): freeze backbone to cut gradient memory; run BN in eval
    if getattr(config, 'LOW_MEM_MODE', False):
        for p in model.feature_extractor.parameters():
            p.requires_grad = False
        model.feature_extractor.eval()
    
    # Data loaders
    train_loader, val_loader = get_data_loaders(config, fold, 'phase1')
    
    # Enhanced wandb initialization with comprehensive tracking (after data loaders)
    if WANDB_AVAILABLE:
        wandb_config = {
            **vars(config),
            'phase': 'Phase_1_ROI_Supervised',
            'model_architecture': 'AttentionMIL',
            'backbone': 'resnet50',
            'total_slides': len(set(train_loader.dataset.slide_paths + val_loader.dataset.slide_paths)),
            'train_slides': len(train_loader.dataset.slide_paths),
            'val_slides': len(val_loader.dataset.slide_paths),
            'augmentation_config': {
                'elastic_deformation_prob': config.ELASTIC_DEFORM_PROB,
                'stain_augment_prob': config.STAIN_AUGMENT_PROB,
                'use_otsu_tissue_mask': config.USE_OTSU_TISSUE_MASK
            }
        }
        
        wandb.init(
            project=config.WANDB_PROJECT, 
            name=f"Phase1_ROI_Supervised_fold{fold}",
            config=wandb_config,
            tags=['phase1', 'roi-supervised', 'classification', f'fold{fold}'],
            notes="ROI-supervised classification training with advanced augmentation"
        )
    
    # Enhanced wandb model tracking
    if WANDB_AVAILABLE:
        # Watch model for gradient and parameter tracking
        # Disable graph logging to save memory; log gradients periodically
        wandb.watch(model, log='gradients', log_freq=50, log_graph=False)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/model_size_mb": total_params * 4 / 1024 / 1024  # Assuming float32
        })
    
    # Optimization: torch.compile for faster training (PyTorch 2.0+)
    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        try:
            # Test compilation with dummy input first
            dummy_input = torch.randn(1, 1, 3, config.PATCH_SIZE, config.PATCH_SIZE).to(config.DEVICE)
            with torch.no_grad():
                _ = model(dummy_input)
            
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled with torch.compile for faster training")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
            # Continue without compilation
    
    # Optimization: Enable cuDNN benchmark for faster convolution
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Ensure checkpoint directory exists
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_PHASE1, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Checkpoint paths
    latest_ckpt_path = config.CHECKPOINT_DIR / f"phase1_fold{fold}_latest.pth"
    best_ckpt_path = config.CHECKPOINT_DIR / f"phase1_fold{fold}_best.pth"

    # Training loop setup with resume support
    best_auc = 0.0
    patience_counter = 0
    global_step = 0
    start_epoch = 0

    # Resume if latest checkpoint exists
    if latest_ckpt_path.exists():
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(ckpt.get('model', {}))
            optimizer.load_state_dict(ckpt.get('optimizer', {}))
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception:
                    pass
            best_auc = ckpt.get('best_auc', best_auc)
            patience_counter = ckpt.get('patience_counter', patience_counter)
            global_step = ckpt.get('global_step', global_step)
            start_epoch = ckpt.get('epoch', -1) + 1
            print(f"[Resume] Loaded checkpoint: {latest_ckpt_path} (epoch {start_epoch}, best AUC {best_auc:.4f})")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint {latest_ckpt_path}: {e}. Starting from scratch.")

    ran_any_epoch = False
    for epoch in range(start_epoch, config.EPOCHS_PHASE1):
        ran_any_epoch = True
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False)):
            images, labels = batch
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # For Phase 1, process each patch individually
                # Reshape for MIL: [batch_size, 1, C, H, W] (1 patch per bag)
                batch_size = images.size(0)
                # Downscale to reduce memory if configured
                input_size = getattr(config, 'INPUT_SIZE', None)
                if input_size is not None and (images.shape[-1] != input_size or images.shape[-2] != input_size):
                    images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False)
                images = images.unsqueeze(1)  # [batch_size, 1, C, H, W]
                
                outputs, attention_weights = model(images)
                # outputs shape: [batch_size, num_classes]
                loss = criterion(outputs, labels)
            
            # Optimization: Gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Optimization: Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            global_step += 1
            
            # Log training metrics every 10 batches
            if batch_idx % 10 == 0:
                if writer: writer.add_scalar('train/batch_loss', loss.item() * config.GRADIENT_ACCUMULATION_STEPS, global_step)
                if writer: writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Enhanced wandb batch-level logging
                if WANDB_AVAILABLE:
                    batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
                    wandb.log({
                        'batch/loss': loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
                        'batch/accuracy': batch_acc,
                        'batch/learning_rate': optimizer.param_groups[0]['lr'],
                        'batch/step': global_step,
                        'batch/epoch': epoch,
                        'batch/samples_processed': global_step * config.BATCH_SIZE
                    }, step=global_step)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   E{epoch+1}", leave=False):
                images, labels = batch
                images = images.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)
                
                # For Phase 1, process each patch individually
                batch_size = images.size(0)
                # Downscale to reduce memory if configured
                input_size = getattr(config, 'INPUT_SIZE', None)
                if input_size is not None and (images.shape[-1] != input_size or images.shape[-2] != input_size):
                    images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=False)
                images = images.unsqueeze(1)  # [batch_size, 1, C, H, W]
                
                with torch.cuda.amp.autocast():
                    outputs, attention_weights = model(images)
                    # outputs shape: [batch_size, num_classes]
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Calculate validation accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_preds)
        
        # Calculate additional metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        val_preds_binary = (np.array(all_preds) > 0.5).astype(int)
        f1 = f1_score(all_labels, val_preds_binary)
        precision = precision_score(all_labels, val_preds_binary)
        recall = recall_score(all_labels, val_preds_binary)
        
        # Early stopping
        if val_auc > best_auc + config.MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            model_path = best_ckpt_path
            torch.save(model.state_dict(), model_path)
            
            # Log model artifact to wandb
            if WANDB_AVAILABLE:
                # Create model artifact
                model_artifact = wandb.Artifact(
                    name=f"phase1_model_fold{fold}",
                    type="model",
                    description=f"Best Phase 1 model (AUC: {val_auc:.4f}) at epoch {epoch+1}",
                    metadata={
                        "epoch": epoch + 1,
                        "val_auc": val_auc,
                        "val_accuracy": val_acc,
                        "val_f1": f1,
                        "architecture": "AttentionMIL",
                        "backbone": "resnet50"
                    }
                )
                model_artifact.add_file(str(model_path))
                wandb.log_artifact(model_artifact)
                
                # Also log as checkpoint
                wandb.log({
                    "checkpoint_saved": True,
                    "best_model_auc": val_auc,
                    "best_model_epoch": epoch + 1
                })
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step()

        # Save latest checkpoint for resume
        latest_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_auc': best_auc,
            'patience_counter': patience_counter,
            'global_step': global_step,
        }
        try:
            torch.save(latest_state, latest_ckpt_path)
        except Exception as e:
            print(f"[Checkpoint] Failed to save latest checkpoint: {e}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('train/epoch_accuracy', train_acc, epoch)
            writer.add_scalar('val/epoch_loss', val_loss, epoch)
            writer.add_scalar('val/epoch_accuracy', val_acc, epoch)
            writer.add_scalar('val/epoch_auc', val_auc, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log attention weights histogram
        if writer and attention_weights is not None:
            writer.add_histogram('attention_weights', attention_weights.detach().cpu(), epoch)
        
        # Log model parameters histogram (every 10 epochs)
        if writer and epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'parameters/{name}', param.detach().cpu(), epoch)
        
        # W&B logging with comprehensive metrics
        if WANDB_AVAILABLE:
            # Basic metrics
            wandb_metrics = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'train/correct': train_correct,
                'train/total': train_total,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'val/auc': val_auc,
                'val/f1': f1,
                'val/precision': precision,
                'val/recall': recall,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'best_auc': best_auc,
                'patience_counter': patience_counter
            }
            
            # Log confusion matrix and classification report
            if epoch % 5 == 0:  # Every 5 epochs
                from sklearn.metrics import confusion_matrix, classification_report
                cm = confusion_matrix(all_labels, (np.array(all_preds) > 0.5).astype(int))
                
                # Create confusion matrix plot
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title(f'Confusion Matrix - Epoch {epoch+1}')
                
                wandb_metrics['confusion_matrix'] = wandb.Image(fig)
                plt.close(fig)
                
                # Classification report
                cr = classification_report(all_labels, (np.array(all_preds) > 0.5).astype(int), output_dict=True)
                wandb_metrics.update({
                    'classification/her2_neg_precision': cr['0']['precision'],
                    'classification/her2_neg_recall': cr['0']['recall'],
                    'classification/her2_neg_f1': cr['0']['f1-score'],
                    'classification/her2_pos_precision': cr['1']['precision'],
                    'classification/her2_pos_recall': cr['1']['recall'],
                    'classification/her2_pos_f1': cr['1']['f1-score'],
                    'classification/macro_avg_f1': cr['macro avg']['f1-score'],
                    'classification/weighted_avg_f1': cr['weighted avg']['f1-score']
                })
            
            # Log model weights and gradients (every 10 epochs)
            if epoch % 10 == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb_metrics[f'gradients/{name}'] = wandb.Histogram(param.grad.detach().cpu().numpy())
                    wandb_metrics[f'parameters/{name}'] = wandb.Histogram(param.detach().cpu().numpy())
            
            # Log attention weights if available
            if attention_weights is not None and epoch % 5 == 0:
                wandb_metrics['attention_weights'] = wandb.Histogram(attention_weights.detach().cpu().numpy())
            
            wandb.log(wandb_metrics)
        
    if ran_any_epoch:
        # Print summary for the last epoch executed
        try:
            elapsed = time.time() - epoch_start
            remaining_epochs = config.EPOCHS_PHASE1 - (epoch + 1)
            eta_sec = elapsed * max(remaining_epochs, 0)
            eta_min = eta_sec / 60.0 if eta_sec > 0 else 0.0
            print(f"Epoch {epoch+1}/{config.EPOCHS_PHASE1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m")
        except Exception:
            pass
    else:
        print("[Resume] Phase 1: No epochs to run (already completed).")

    if writer:
        writer.close()
    if WANDB_AVAILABLE and ran_any_epoch:
        # Log final training summary
        wandb.log({
            "training_complete": True,
            "final_best_auc": best_auc,
            "total_epochs_trained": epoch + 1,
            "early_stopped": patience_counter >= config.PATIENCE
        })
        
        # Create and log training summary table
        summary_data = [
            ["Metric", "Value"],
            ["Best Validation AUC", f"{best_auc:.4f}"],
            ["Final Validation Accuracy", f"{val_acc:.2f}%"],
            ["Final F1 Score", f"{f1:.4f}"],
            ["Final Precision", f"{precision:.4f}"],
            ["Final Recall", f"{recall:.4f}"],
            ["Total Epochs", epoch + 1],
            ["Early Stopped", patience_counter >= config.PATIENCE]
        ]
        wandb.log({"training_summary": wandb.Table(data=summary_data, columns=["Metric", "Value"])})
        
    wandb.finish()

def train_phase2(config: Config, fold: int = 0, writer = None):
    """Phase 2: MIL fine-tuning with frozen early layers"""
    
    if WANDB_AVAILABLE:
        # Enhanced wandb initialization for Phase 2
        wandb_config = {
            **vars(config),
            'phase': 'Phase_2_MIL_FineTuning',
            'model_architecture': 'AttentionMIL_FineTuned',
            'strategy': 'Frozen_Backbone_FineTune_Attention',
            'pretrained_from': 'Phase1_ROI_Supervised'
        }
        
        wandb.init(
            project=config.WANDB_PROJECT, 
            name=f"Phase2_MIL_FineTuning_fold{fold}",
            config=wandb_config,
            tags=['phase2', 'mil-finetuning', 'attention', f'fold{fold}'],
            notes="MIL fine-tuning with frozen backbone, training attention mechanism"
        )
    
    # Initialize TensorBoard writer if available and not provided
    if writer is None and TENSORBOARD_AVAILABLE and config.TENSORBOARD_DIR:
        writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / f"phase2_fold{fold}"))
    
    # Load pre-trained model from Phase 1
    model = AttentionMIL(backbone='resnet50', num_classes=config.NUM_CLASSES,
                        attention_dim=config.ATTENTION_DIM, dropout=config.DROPOUT_RATE)
    
    # Load Phase 1 weights
    phase1_path = config.CHECKPOINT_DIR / f"phase1_fold{fold}_best.pth"
    if phase1_path.exists():
        model.load_state_dict(torch.load(phase1_path, map_location='cpu'))
        print(f"Loaded Phase 1 weights from {phase1_path}")
    else:
        print("Warning: Phase 1 weights not found, starting from scratch")
    
    model = model.to(config.DEVICE)
    
    # Freeze early layers of the feature extractor
    for name, param in model.feature_extractor.named_parameters():
        if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
            param.requires_grad = False
            print(f"Frozen layer: {name}")
    
    # Optimization: torch.compile for faster training
    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        try:
            # Test compilation first
            dummy_input = torch.randn(1, 1, 3, config.PATCH_SIZE, config.PATCH_SIZE).to(config.DEVICE)
            with torch.no_grad():
                _ = model(dummy_input)
            
            model = torch.compile(model, mode='reduce-overhead')
            print("Phase 2 model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed for Phase 2 model: {e}")
    
    # Optimization: Enable cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Ensure checkpoint directory exists
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Use lower learning rate for fine-tuning
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.LR_PHASE2, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Use Focal Loss for better handling of hard examples
    if hasattr(monai.losses, 'FocalLoss'):
        criterion = monai.losses.FocalLoss(alpha=0.25, gamma=2.0, to_onehot_y=True, include_background=False)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()

    # Checkpoint paths and resume state
    latest_ckpt_path = config.CHECKPOINT_DIR / f"phase2_fold{fold}_latest.pth"
    best_ckpt_path = config.CHECKPOINT_DIR / f"phase2_fold{fold}_best.pth"
    best_auc = 0.0
    patience_counter = 0
    global_step = 0
    start_epoch = 0

    if latest_ckpt_path.exists():
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(ckpt.get('model', {}))
            optimizer.load_state_dict(ckpt.get('optimizer', {}))
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception:
                    pass
            best_auc = ckpt.get('best_auc', best_auc)
            patience_counter = ckpt.get('patience_counter', patience_counter)
            global_step = ckpt.get('global_step', global_step)
            start_epoch = ckpt.get('epoch', -1) + 1
            print(f"[Resume] Phase 2: Loaded checkpoint {latest_ckpt_path} (epoch {start_epoch}, best AUC {best_auc:.4f})")
        except Exception as e:
            print(f"[Resume] Phase 2: Failed to load checkpoint {latest_ckpt_path}: {e}. Starting from scratch.")
    
    # Create MIL data loaders (more patches per slide)
    temp_config = Config()
    temp_config.__dict__.update(config.__dict__)
    temp_config.PATCHES_PER_SLIDE_PHASE1 = config.PATCHES_PER_SLIDE_PHASE2
    
    # Get datasets with MIL phase
    if not config.METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {config.METADATA_FILE}")
    
    metadata = pd.read_csv(config.METADATA_FILE)
    
    # Get slide paths and labels
    slide_paths = []
    labels = []
    annotations = []
    
    for _, row in metadata.iterrows():
        slide_name = row['slide_name']
        slide_path = config.DATA_DIR / f"{slide_name}.svs"
        
        if slide_path.exists():
            slide_paths.append(str(slide_path))
            labels.append(int(row['her2_status']))
            
            # Find annotation file
            ann_path = config.ANNOTATIONS_DIR / f"{slide_name}.xml"
            annotations.append(str(ann_path) if ann_path.exists() else None)
    
    # Stratified split
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(slide_paths, labels))
    
    train_indices, val_indices = splits[fold]
    
    train_paths = [slide_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_annotations = [annotations[i] for i in train_indices]
    
    val_paths = [slide_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_annotations = [annotations[i] for i in val_indices]
    
    # Stain normalizer using imported function
    stain_normalizer = create_stain_normalizer()
    if stain_normalizer and config.DATA_DIR:
        reference_path = config.DATA_DIR / "reference.png"
        if reference_path.exists():
            reference = Image.open(reference_path)
            stain_normalizer.fit(np.array(reference))
    
    # MIL phase transforms using imported functions
    train_transform = get_classification_transforms('train')
    val_transform = get_classification_transforms('val')
    
    train_dataset = HER2WSIDataset(
        train_paths, train_labels, train_annotations,
        patch_size=config.PATCH_SIZE, transform=train_transform,
        stain_normalizer=stain_normalizer, phase='mil'  # Use MIL phase
    )
    
    val_dataset = HER2WSIDataset(
        val_paths, val_labels, val_annotations,
        patch_size=config.PATCH_SIZE, transform=val_transform,
        stain_normalizer=stain_normalizer, phase='mil'
    )
    
    # Data loaders
    train_loader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': config.NUM_WORKERS > 0
    }
    
    val_loader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': config.NUM_WORKERS > 0
    }
    
    if config.NUM_WORKERS > 0:
        train_loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR
        val_loader_kwargs['prefetch_factor'] = config.PREFETCH_FACTOR
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    ran_any_epoch = False
    for epoch in range(start_epoch, config.EPOCHS_PHASE2):
        ran_any_epoch = True
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train2 E{epoch+1}", leave=False)):
            images, labels = batch
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            
            # MIL: group patches as instances
            batch_size = images.size(0)
            num_instances = min(20, batch_size)  # Limit instances for memory
            images = images[:num_instances].unsqueeze(0)  # [1, num_instances, C, H, W]
            labels = labels[:1]  # Use only first label (slide-level)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs, attention_weights = model(images)
                loss = criterion(outputs, labels)
            
            # Gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            global_step += 1
            
            # Log training metrics
            if batch_idx % 10 == 0 and writer:
                writer.add_scalar('train_phase2/batch_loss', loss.item() * config.GRADIENT_ACCUMULATION_STEPS, global_step)
                writer.add_scalar('train_phase2/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                # Log attention weights
                if attention_weights is not None:
                    writer.add_histogram('attention_weights_phase2', attention_weights.detach().cpu(), global_step)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val2   E{epoch+1}", leave=False):
                images, labels = batch
                images = images.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)
                
                batch_size = images.size(0)
                num_instances = min(20, batch_size)
                images = images[:num_instances].unsqueeze(0)
                labels = labels[:1]
                
                with torch.cuda.amp.autocast():
                    outputs, attention_weights = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Calculate validation accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        
        # Early stopping
        if val_auc > best_auc + config.MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step()

        # Save latest checkpoint for resume
        latest_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_auc': best_auc,
            'patience_counter': patience_counter,
            'global_step': global_step,
        }
        try:
            torch.save(latest_state, latest_ckpt_path)
        except Exception as e:
            print(f"[Checkpoint] Phase 2: Failed to save latest checkpoint: {e}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('train_phase2/epoch_loss', train_loss, epoch)
            writer.add_scalar('train_phase2/epoch_accuracy', train_acc, epoch)
            writer.add_scalar('val_phase2/epoch_loss', val_loss, epoch)
            writer.add_scalar('val_phase2/epoch_accuracy', val_acc, epoch)
            writer.add_scalar('val_phase2/epoch_auc', val_auc, epoch)
            writer.add_scalar('train_phase2/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # W&B logging
        if WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss_phase2': train_loss,
                'train_accuracy_phase2': train_acc,
                'val_loss_phase2': val_loss,
                'val_accuracy_phase2': val_acc,
                'val_auc_phase2': val_auc,
                'lr_phase2': optimizer.param_groups[0]['lr']
            })
        
    if ran_any_epoch:
        try:
            elapsed = time.time() - epoch_start
            eta_min = (elapsed * (config.EPOCHS_PHASE2 - (epoch + 1))) / 60.0
            print(f"Phase 2 Epoch {epoch+1}/{config.EPOCHS_PHASE2} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m")
        except Exception:
            pass
    else:
        print("[Resume] Phase 2: No epochs to run (already completed).")

    if writer:
        writer.close()
    if WANDB_AVAILABLE:
        wandb.finish()

def explain_predictions(config: Config, model_path: str, fold: int = 0, num_samples: int = 5):
    """Generate explanations using Grad-CAM and attention maps"""
    if not CAM_AVAILABLE:
        print("PyTorch Grad-CAM not available, skipping explainability")
        return
    
    print("Generating explanations...")
    
    # Load model
    model = AttentionMIL(backbone='resnet50', num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Get validation data
    _, val_loader = get_data_loaders(config, fold, 'phase1')

    # Wrap MIL model so CAM works on [N, C, H, W]
    class _MILWrapper(nn.Module):
        def __init__(self, mil: AttentionMIL):
            super().__init__()
            self.mil = mil
            self.feature_extractor = mil.feature_extractor
        def forward(self, x):
            x = x.unsqueeze(1)  # [N, 1, C, H, W]
            logits, _ = self.mil(x)
            return logits
    wrapper_model = _MILWrapper(model).to(config.DEVICE)
    wrapper_model.eval()

    # Select a robust target layer
    fe = wrapper_model.feature_extractor
    if hasattr(fe, 'layer4'):
        try:
            target_layer = fe.layer4[-1]
        except Exception:
            target_layer = fe.layer4
    elif hasattr(fe, 'features'):
        try:
            target_layer = fe.features[-1]
        except Exception:
            target_layer = fe.features
    else:
        target_layer = list(fe.modules())[-1]

    cam = GradCAM(model=wrapper_model, target_layers=[target_layer])
    
    explanation_dir = config.LOG_DIR / "explanations" / f"fold{fold}"
    explanation_dir.mkdir(parents=True, exist_ok=True)
    
    sample_count = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if sample_count >= num_samples:
            break

        images, labels = batch
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        with torch.no_grad():
            logits = wrapper_model(images)
            preds = torch.argmax(logits, dim=1)

        for i in range(min(images.size(0), 5)):
            if sample_count >= num_samples:
                break

            input_tensor = images[i:i+1]
            targets = [ClassifierOutputTarget(int(preds[i].item()))]

            with torch.enable_grad():
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            img_np = images[i].detach().cpu().permute(1, 2, 0).numpy()
            denom = (img_np.max() - img_np.min()) + 1e-8
            img_np = (img_np - img_np.min()) / denom

            visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title(f"Original\nTrue: {labels[i].item()}, Pred: {preds[i].item()}")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(grayscale_cam, cmap='hot')
            plt.title("Grad-CAM")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(visualization)
            plt.title("Overlay")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(explanation_dir / f"sample_{sample_count}.png", dpi=150, bbox_inches='tight')
            plt.close()

            sample_count += 1
    
    print(f"Saved {sample_count} explanation samples to {explanation_dir}")

def train_segmentation(config: Config, fold: int = 0, writer = None):
    """Train segmentation model"""
    
    if WANDB_AVAILABLE:
        wandb.init(project=config.WANDB_PROJECT, name=f"segmentation_fold{fold}", config=vars(config))
    
    # Initialize TensorBoard writer if available and not provided
    if writer is None and TENSORBOARD_AVAILABLE and config.TENSORBOARD_DIR:
        writer = SummaryWriter(log_dir=str(config.TENSORBOARD_DIR / f"segmentation_fold{fold}"))
    
    # Ensure checkpoint directory exists
    try:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Model
    model = SegmentationUNet()
    model = model.to(config.DEVICE)
    
    # Optimization: torch.compile for faster training
    if config.USE_TORCH_COMPILE and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Segmentation model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")
    
    # Optimization: Enable cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_PHASE1, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Loss
    dice_loss = DiceLoss(sigmoid=True)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # AMP
    scaler = torch.cuda.amp.GradScaler()
    
    # Checkpoint paths and resume state
    latest_ckpt_path = config.CHECKPOINT_DIR / f"segmentation_fold{fold}_latest.pth"
    best_ckpt_path = config.CHECKPOINT_DIR / f"segmentation_fold{fold}_best.pth"
    best_dice = 0.0
    patience_counter = 0
    global_step = 0
    start_epoch = 0

    if latest_ckpt_path.exists():
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=config.DEVICE)
            model.load_state_dict(ckpt.get('model', {}))
            optimizer.load_state_dict(ckpt.get('optimizer', {}))
            if 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
            if 'scaler' in ckpt:
                try:
                    scaler.load_state_dict(ckpt['scaler'])
                except Exception:
                    pass
            best_dice = ckpt.get('best_dice', best_dice)
            patience_counter = ckpt.get('patience_counter', patience_counter)
            global_step = ckpt.get('global_step', global_step)
            start_epoch = ckpt.get('epoch', -1) + 1
            print(f"[Resume] Seg: Loaded checkpoint {latest_ckpt_path} (epoch {start_epoch}, best Dice {best_dice:.4f})")
        except Exception as e:
            print(f"[Resume] Seg: Failed to load checkpoint {latest_ckpt_path}: {e}. Starting from scratch.")

    # Data loaders with optimizations
    train_loader, val_loader = get_data_loaders(config, fold, 'segmentation')
    
    ran_any_epoch = False
    for epoch in range(start_epoch, config.EPOCHS_PHASE1):
        ran_any_epoch = True
        epoch_start = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_dice_scores = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"TrainSeg E{epoch+1}", leave=False)):
            images, masks = batch
            images = images.to(config.DEVICE, non_blocking=True)
            masks = masks.to(config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                dice_l = dice_loss(outputs, masks)
                bce_l = bce_loss(outputs, masks)
                loss = dice_l + bce_l
            
            # Optimization: Gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Optimization: Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # Calculate training Dice
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                dice = (2 * (preds * masks).sum() / (preds + masks).sum()).item()
                train_dice_scores.append(dice)
            
            global_step += 1
            
            # Log training metrics every 10 batches
            if batch_idx % 10 == 0 and writer:
                writer.add_scalar('train/batch_loss', loss.item() * config.GRADIENT_ACCUMULATION_STEPS, global_step)
                writer.add_scalar('train/batch_dice', dice, global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        train_loss /= len(train_loader)
        train_dice = np.mean(train_dice_scores)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        val_iou_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"ValSeg   E{epoch+1}", leave=False):
                images, masks = batch
                images = images.to(config.DEVICE, non_blocking=True)
                masks = masks.to(config.DEVICE, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    dice_l = dice_loss(outputs, masks)
                    bce_l = bce_loss(outputs, masks)
                    loss = dice_l + bce_l
                
                val_loss += loss.item()
                
                # Calculate validation metrics
                preds = torch.sigmoid(outputs) > 0.5
                dice = (2 * (preds * masks).sum() / (preds + masks).sum()).item()
                iou = (preds * masks).sum() / ((preds + masks).sum() - (preds * masks).sum()).item()
                
                val_dice_scores.append(dice)
                val_iou_scores.append(iou)
        
        val_loss /= len(val_loader)
        val_dice = np.mean(val_dice_scores)
        val_iou = np.mean(val_iou_scores)
        
        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1
        
        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
        
        scheduler.step()

        # Save latest checkpoint for resume
        latest_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_dice': best_dice,
            'patience_counter': patience_counter,
            'global_step': global_step,
        }
        try:
            torch.save(latest_state, latest_ckpt_path)
        except Exception as e:
            print(f"[Checkpoint] Seg: Failed to save latest checkpoint: {e}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
            writer.add_scalar('train/epoch_dice', train_dice, epoch)
            writer.add_scalar('val/epoch_loss', val_loss, epoch)
            writer.add_scalar('val/epoch_dice', val_dice, epoch)
            writer.add_scalar('val/epoch_iou', val_iou, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log model parameters histogram (every 10 epochs)
        if writer and epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'parameters/{name}', param.detach().cpu(), epoch)
        
        # Log sample predictions (every 5 epochs)
        if writer and epoch % 5 == 0:
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images[:4].to(config.DEVICE)
                sample_masks = sample_masks[:4].to(config.DEVICE)
                
                with torch.cuda.amp.autocast():
                    sample_outputs = model(sample_images)
                
                sample_preds = torch.sigmoid(sample_outputs) > 0.5
                
                # Log images
                writer.add_images('validation/images', sample_images, epoch)
                writer.add_images('validation/masks', sample_masks, epoch)
                writer.add_images('validation/predictions', sample_preds.float(), epoch)
        
        # W&B logging
        if WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'lr': optimizer.param_groups[0]['lr']
            })
        
    if ran_any_epoch:
        try:
            elapsed = time.time() - epoch_start
            eta_min = (elapsed * (config.EPOCHS_PHASE1 - (epoch + 1))) / 60.0
            print(f"Epoch {epoch+1}/{config.EPOCHS_PHASE1} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m")
        except Exception:
            pass
    else:
        print("[Resume] Segmentation: No epochs to run (already completed).")

    if writer:
        writer.close()
    if WANDB_AVAILABLE:
        wandb.finish()

def optimize_hyperparameters(config: Config, n_trials: int = 50):
    """Bayesian hyperparameter optimization using Optuna"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, skipping hyperparameter optimization")
        return config
    
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
        dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
        attention_dim = trial.suggest_categorical('attention_dim', [64, 128, 256])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Create temporary config
        temp_config = Config()
        temp_config.__dict__.update(config.__dict__)
        temp_config.LR_PHASE1 = lr
        temp_config.BATCH_SIZE = batch_size
        temp_config.DROPOUT_RATE = dropout
        temp_config.ATTENTION_DIM = attention_dim
        temp_config.WEIGHT_DECAY = weight_decay
        temp_config.EPOCHS_PHASE1 = 10  # Shorter training for optimization
        
        try:
            # Quick training
            best_auc = train_phase1_for_optuna(temp_config, fold=0)
            return best_auc
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    # Create study
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler())
    
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)
    
    # Update config with best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best AUC: {study.best_value:.4f}")
    
    # Update config
    for key, value in best_params.items():
        if key == 'learning_rate':
            config.LR_PHASE1 = value
            config.LEARNING_RATE = value
        elif key == 'batch_size':
            config.BATCH_SIZE = value
        elif key == 'dropout_rate':
            config.DROPOUT_RATE = value
        elif key == 'attention_dim':
            config.ATTENTION_DIM = value
        elif key == 'weight_decay':
            config.WEIGHT_DECAY = value
    
    return config

def train_phase1_for_optuna(config: Config, fold: int = 0):
    """Simplified Phase 1 training for Optuna optimization"""
    # Similar to train_phase1 but simplified and returns best AUC
    model = AttentionMIL(backbone='resnet50', num_classes=config.NUM_CLASSES,
                        attention_dim=config.ATTENTION_DIM, dropout=config.DROPOUT_RATE)
    model = model.to(config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LR_PHASE1, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    train_loader, val_loader = get_data_loaders(config, fold, 'phase1')
    
    best_auc = 0.0
    
    for epoch in range(config.EPOCHS_PHASE1):
        # Training
        model.train()
        for batch in train_loader:
            images, labels = batch
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            images = images.unsqueeze(0)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs, _ = model(images)
                loss = criterion(outputs.squeeze(0), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True)
                images = images.unsqueeze(0)
                
                with torch.cuda.amp.autocast():
                    outputs, _ = model(images)
                
                preds = torch.softmax(outputs.squeeze(0), dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            if auc > best_auc:
                best_auc = auc
    
    return best_auc

def main():
    parser = argparse.ArgumentParser(description="HER2 Breast Cancer Classification and Segmentation")
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--fold', type=int, default=0, help='Cross-validation fold')
    parser.add_argument('--phase', type=str, choices=['phase1', 'phase2', 'segmentation', 'all', 'optimize'], default='all', help='Training phase')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--optimize-trials', type=int, default=50, help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    config = Config()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create directories
    config.LOG_DIR.mkdir(exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    config.TENSORBOARD_DIR.mkdir(exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(str(config.TENSORBOARD_DIR))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_DIR / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Log hyperparameters
    hparams = {
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'max_epochs': config.MAX_EPOCHS,
        'gradient_accumulation_steps': config.GRADIENT_ACCUMULATION_STEPS,
        'max_grad_norm': config.MAX_GRAD_NORM,
        'use_torch_compile': config.USE_TORCH_COMPILE
    }
    writer.add_hparams(hparams, {})
    
    print("Starting HER2+ Breast Cancer Classification and Segmentation Pipeline...")
    print(f"TensorBoard logs: {config.TENSORBOARD_DIR}")
    
    try:
        # Hyperparameter optimization
        if args.phase == 'optimize':
            print("Starting hyperparameter optimization...")
            config = optimize_hyperparameters(config, args.optimize_trials)
            # Save optimized config
            optimized_config = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            with open('optimized_config.json', 'w') as f:
                json.dump(optimized_config, f, indent=2, default=str)
            print("Optimized configuration saved to optimized_config.json")
        
        # Train
        if args.phase in ['phase1', 'all']:
            print("Starting Phase 1: ROI-supervised classification")
            train_phase1(config, args.fold, writer)
            
            # Generate explanations
            phase1_model = config.CHECKPOINT_DIR / f"phase1_fold{args.fold}_best.pth"
            if phase1_model.exists():
                explain_predictions(config, str(phase1_model), args.fold)
        
        if args.phase in ['phase2', 'all']:
            print("Starting Phase 2: MIL fine-tuning")
            train_phase2(config, args.fold, writer)
        
        if args.phase in ['segmentation', 'all']:
            print("Starting Segmentation training")
            train_segmentation(config, args.fold, writer)
        
        print("Training completed successfully!")
        print(f"View TensorBoard logs: tensorboard --logdir {config.TENSORBOARD_DIR}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        writer.close()

if __name__ == '__main__':
    main()