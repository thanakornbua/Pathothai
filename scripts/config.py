"""
Configuration Management for HER2+ Breast Cancer Pipeline

This module provides centralized configuration management for the entire pipeline.

Authors:
    - Primary: T. Buathongtanakarn
    - AI Assistant: GitHub Copilot

Version: 2.1.0
Last Updated: September 17, 2025
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DataConfig:
    """Data configuration parameters"""
    # Paths
    data_dir: str = "data"
    annotations_dir: str = "Annotations"
    output_dir: str = "output"
    checkpoints_dir: str = "checkpoints"
    
    # Data parameters
    patch_size: int = 224
    tissue_threshold: float = 0.01
    overlap_factor: float = 0.0
    
    # WSI processing
    level: int = 0
    thumbnail_size: Tuple[int, int] = (512, 512)
    
    def __post_init__(self):
        """Ensure paths exist"""
        for path_attr in ['data_dir', 'annotations_dir', 'output_dir', 'checkpoints_dir']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Architecture
    model_name: str = "resnet18"
    num_classes: int = 2
    pretrained: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 20
    
    # Optimization
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    
    # Loss function
    loss_function: str = "cross_entropy"
    class_weights: Optional[List[float]] = None
    
    # MIL specific
    mil_pooling: str = "attention"
    mil_hidden_dim: int = 256
    
    # Segmentation specific
    segmentation_classes: int = 3
    segmentation_architecture: str = "unet"


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Phases
    enable_classification: bool = True
    enable_segmentation: bool = True
    enable_mil: bool = True
    enable_gradcam: bool = True
    
    # Validation
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Checkpointing
    save_best_only: bool = True
    save_frequency: int = 10
    
    # Logging
    log_frequency: int = 10
    tensorboard_logging: bool = True
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True


@dataclass
class AugmentationConfig:
    """Augmentation configuration parameters"""
    # General
    augment_probability: float = 0.5
    
    # Geometric transforms
    rotation_degrees: int = 45
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    
    # Color transforms
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    
    # Noise and blur
    gaussian_noise_std: float = 0.02
    gaussian_blur_prob: float = 0.2
    
    # Stain augmentation
    he_augmentation: bool = True
    stain_augment_prob: float = 0.3
    
    # Elastic deformation (segmentation)
    elastic_deform_prob: float = 0.2
    elastic_deform_alpha: float = 100.0
    elastic_deform_sigma: float = 10.0
    
    # Normalization
    stain_normalization: bool = False
    normalization_method: str = "macenko"


@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration"""
    # Optuna settings
    n_trials: int = 100
    study_name: str = "her2_optimization"
    storage_url: Optional[str] = None
    
    # Search spaces
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [16, 32, 64])
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-2)
    
    # Pruning
    enable_pruning: bool = True
    pruning_patience: int = 10
    
    # Objectives
    primary_metric: str = "f1_score"
    direction: str = "maximize"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Sub-configurations
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    augmentation: AugmentationConfig = None
    optimization: OptimizationConfig = None
    
    # Metadata
    version: str = "2.1.0"
    experiment_name: str = "her2_experiment"
    description: str = "HER2+ Breast Cancer Classification Pipeline"
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        config_dict = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, default=str)
        print(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        data_config = DataConfig(**config_dict.pop('data', {}))
        model_config = ModelConfig(**config_dict.pop('model', {}))
        training_config = TrainingConfig(**config_dict.pop('training', {}))
        augmentation_config = AugmentationConfig(**config_dict.pop('augmentation', {}))
        optimization_config = OptimizationConfig(**config_dict.pop('optimization', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            augmentation=augmentation_config,
            optimization=optimization_config,
            **config_dict
        )
    
    def get_config_summary(self) -> str:
        """Get a formatted summary of the configuration"""
        summary = f"""
📋 Pipeline Configuration Summary
{'=' * 50}
Experiment: {self.experiment_name}
Version: {self.version}
Description: {self.description}

🗂️ Data Configuration:
  - Patch size: {self.data.patch_size}
  - Data directory: {self.data.data_dir}
  - Tissue threshold: {self.data.tissue_threshold}

🧠 Model Configuration:
  - Architecture: {self.model.model_name}
  - Classes: {self.model.num_classes}
  - Batch size: {self.model.batch_size}
  - Learning rate: {self.model.learning_rate}

🎯 Training Configuration:
  - Epochs: {self.model.num_epochs}
  - Early stopping: {self.training.early_stopping_patience}
  - Validation split: {self.training.validation_split}
  - Device: {self.training.device}

🎨 Augmentation Configuration:
  - H&E augmentation: {self.augmentation.he_augmentation}
  - Rotation degrees: {self.augmentation.rotation_degrees}
  - Augmentation probability: {self.augmentation.augment_probability}

🔧 Optimization Configuration:
  - Trials: {self.optimization.n_trials}
  - Primary metric: {self.optimization.primary_metric}
  - Pruning enabled: {self.optimization.enable_pruning}
"""
        return summary.strip()


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration"""
    return PipelineConfig(
        experiment_name="her2_default_experiment",
        description="Default HER2+ classification pipeline configuration"
    )


def create_quick_test_config() -> PipelineConfig:
    """Create a configuration for quick testing"""
    config = create_default_config()
    config.experiment_name = "her2_quick_test"
    config.description = "Quick test configuration with reduced parameters"
    
    # Reduce for faster testing
    config.model.num_epochs = 5
    config.model.batch_size = 16
    config.training.early_stopping_patience = 3
    config.optimization.n_trials = 10
    
    return config


def create_production_config() -> PipelineConfig:
    """Create a configuration for production training"""
    config = create_default_config()
    config.experiment_name = "her2_production"
    config.description = "Production configuration with optimized parameters"
    
    # Production settings
    config.model.num_epochs = 15
    config.model.batch_size = 16
    config.training.early_stopping_patience = 5
    config.optimization.n_trials = 200
    config.augmentation.augment_probability = 0.7
    
    return config


if __name__ == "__main__":
    # Example usage
    print("🔧 Configuration Management Examples")
    print("=" * 50)
    
    # Create default configuration
    config = create_default_config()
    print(config.get_config_summary())
    
    # Save configuration
    config.save_config("config_default.json")
    
    # Load configuration
    loaded_config = PipelineConfig.load_config("config_default.json")
    print(f"\n✅ Configuration loaded successfully: {loaded_config.experiment_name}")
    
    # Create different configurations
    test_config = create_quick_test_config()
    production_config = create_production_config()
    
    print(f"\n📊 Available configurations:")
    print(f"  - Default: {config.experiment_name}")
    print(f"  - Quick test: {test_config.experiment_name}")
    print(f"  - Production: {production_config.experiment_name}")
