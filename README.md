# HER2+ Breast Cancer Classification and Segmentation Pipeline

A comprehensive deep learning pipeline for automated HER2+ breast cancer classification and segmentation from H&E-stained whole-slide images (WSIs).

## 🎯 Overview

This pipeline implements a multi-phase training approach:
1. **Phase 1**: ROI-supervised classification
2. **Phase 2**: Multiple Instance Learning (MIL) 
3. **Segmentation**: Pixel-level tumor segmentation

## ✨ Key Features

- **Multi-phase Training**: Progressive learning from ROI annotations to whole-slide analysis
- **Advanced Augmentation**: H&E stain variation, elastic deformation, tissue-aware sampling
- **GPU Optimization**: Mixed precision training, PyTorch 2.0 compilation
- **Comprehensive Evaluation**: ROC-AUC, confusion matrices, attention maps
- **Model Explainability**: Grad-CAM visualizations, attention analysis
- **Hyperparameter Optimization**: Optuna-based automated tuning

## Requirements

### Software Dependencies

```bash
pip install torch torchvision torchaudio
pip install monai
pip install openslide-python
pip install staintools
pip install wandb
pip install optuna
pip install pytorch-grad-cam
pip install pandas scikit-learn matplotlib seaborn
```

### Hardware Requirements

- **GPU**: NVIDIA A100 or RTX 4090 (recommended)
- **RAM**: 32GB+ for large WSIs
- **Storage**: 500GB+ for dataset and checkpoints

## Jupyter Notebook Usage

The training pipeline can be used as a Python module in Jupyter notebooks for interactive development and experimentation.

### Quick Start

```python
# Import the training module
from scripts.train import quick_train, get_training_summary, setup_notebook_logging

# Setup logging for notebooks
setup_notebook_logging()

# Quick training
trainer = quick_train(
    data_dir='data/train',
    model_name='resnet50',
    num_classes=2,
    num_epochs=10,
    task='mtl'  # Multi-task learning
)

# View training summary
print(get_training_summary(trainer))
```

### Advanced Usage

```python
from scripts.train import WSITrainer, plot_training_history

# Create custom trainer
trainer = WSITrainer(
    model_name='resnet50',
    num_classes=2,
    task='mtl',
    checkpoint_dir='checkpoints/custom'
)

# Prepare data
trainer.prepare_data('data/train', batch_size=16, patch_size=256)

# Setup training
trainer.setup_training(learning_rate=0.001)

# Train
trainer.train(num_epochs=20)

# Visualize results
plot_training_history(trainer)
```

### Notebook Features

- **Real-time Progress**: Training progress with timestamps and emojis
- **Interactive Monitoring**: Live updates during training
- **Easy Configuration**: Simple function calls for common tasks
- **Visualization**: Built-in plotting functions for training history
- **Checkpoint Management**: Automatic saving and resuming
- **TensorBoard Integration**: Web-based monitoring interface

See `HER2_Training_Notebook.ipynb` for a complete example.
├── annotations/
│   ├── HER2Neg_Case_01.xml
│   ├── HER2Pos_Case_01.xml
│   └── ...
├── metadata.csv
└── reference.png  # Reference image for stain normalization
```

### Metadata CSV Format

```csv
slide_name,her2_status,age,sex,tumor_type,treatment_outcome
HER2Neg_Case_01,0,45,F,invasive ductal carcinoma,recurrence
HER2Pos_Case_01,1,52,F,invasive ductal carcinoma,complete response
...
```

### Annotation Format

XML files should contain ROI annotations with the following structure:

```xml
<Annotations>
    <Annotation Name="Tumor">
        <Regions>
            <Region>
                <Vertices>
                    <Vertex X="100" Y="200" />
                    <Vertex X="150" Y="250" />
                    ...
                </Vertices>
            </Region>
        </Regions>
    </Annotation>
</Annotations>
```

## Usage

### 1. Configuration

Edit `config.json` to customize training parameters:

```json
{
    "DATA_DIR": "data",
    "BATCH_SIZE": 16,
    "EPOCHS_PHASE1": 50,
    "LR_PHASE1": 0.0001,
    "PATIENCE": 5
}
```

### 2. Training

#### Phase 1: ROI-supervised Classification

```bash
python train_her2_pipeline.py --phase phase1 --fold 0
```

#### Segmentation Training

```bash
python train_her2_pipeline.py --phase segmentation --fold 0
```

#### Full Pipeline

```bash
python train_her2_pipeline.py --phase all --fold 0
```

### 3. Monitoring Training with TensorBoard

The pipeline includes comprehensive TensorBoard logging for real-time monitoring:

```bash
# Start TensorBoard server
tensorboard --logdir logs/tensorboard

# Open browser to http://localhost:6006
```

**Logged Metrics:**
- Training/validation loss and accuracy
- AUC-ROC scores
- Dice coefficient and IoU for segmentation
- Learning rate schedules
- Model histograms (weights, gradients, activations)
- Sample predictions and attention maps

### 4. Performance Optimizations

The pipeline includes several optimizations for faster training:

- **torch.compile**: JIT compilation for optimized execution
- **Gradient Accumulation**: Larger effective batch sizes
- **cuDNN Benchmark**: Optimized convolution algorithms
- **TF32 Precision**: Faster matrix operations on Ampere GPUs
- **Non-blocking Data Transfer**: Asynchronous GPU transfers
- **Persistent Workers**: Reusable DataLoader workers

Configure optimizations in `config.json`:

```json
{
    "USE_TORCH_COMPILE": true,
    "GRADIENT_ACCUMULATION_STEPS": 2,
    "MAX_GRAD_NORM": 1.0,
    "PREFETCH_FACTOR": 2
}
```

### 5. Cross-validation

Run 5-fold cross-validation:

```bash
for fold in {0..4}; do
    python train_her2_pipeline.py --phase all --fold $fold
done
```

## Model Architectures

### Classification Model

- **Backbone**: ResNet-50 or EfficientNet-B0 (pretrained on ImageNet)
- **Attention Mechanism**: Learnable attention weights for instance aggregation
- **Output**: Slide-level HER2 status prediction

### Segmentation Model

- **Architecture**: U-Net with residual blocks
- **Loss**: Dice Loss + Binary Cross Entropy
- **Output**: Pixel-level tumor mask

## Evaluation Metrics

### Classification
- **AUC-ROC**: Primary metric for HER2 status classification
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-specific performance

### Segmentation
- **Dice Coefficient**: Overlap between predicted and ground truth masks
- **IoU (Jaccard Index)**: Intersection over Union
- **Precision/Recall**: Pixel-level performance

## Explainability

### Grad-CAM Visualization

```python
from train_her2_pipeline import explain_predictions

# Load model and image
model = AttentionMIL()
model.load_state_dict(torch.load('checkpoints/phase1_fold0_best.pth'))

# Generate explanation
cam = explain_predictions(model, image_tensor, target_class=1)
```

### Attention Maps

The MIL model outputs attention weights that can be visualized to show which patches contributed most to the slide-level prediction.

## Results Visualization

### W&B Dashboard

Monitor training progress, metrics, and hyperparameters in real-time:

```bash
wandb login
# Training logs will automatically appear in your W&B project
```

### Local Visualization

```python
# Plot training curves
import matplotlib.pyplot as plt
import pandas as pd

# Load training history
history = pd.read_csv('logs/training_history.csv')

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['val_auc'])
plt.title('Validation AUC')

plt.subplot(1, 3, 3)
plt.plot(history['lr'])
plt.title('Learning Rate')
plt.yscale('log')
plt.tight_layout()
plt.show()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in config.json
   - Use gradient accumulation
   - Enable `torch.cuda.empty_cache()`

2. **Slow Training**
   - Increase `NUM_WORKERS` for data loading
   - Use SSD storage for data
   - Enable cuDNN benchmark: `torch.backends.cudnn.benchmark = True`

3. **Stain Normalization Errors**
   - Ensure reference.png exists and is a valid H&E image
   - Install StainTools: `pip install staintools`

4. **Annotation Loading Errors**
   - Verify XML files follow the expected format
   - Check coordinate ranges match WSI dimensions

### Performance Optimization

- **Mixed Precision**: Enabled by default for faster training
- **Data Loading**: Use multiple workers and pin memory
- **Model Parallelism**: For very large models, use `torch.nn.DataParallel`
- **Gradient Checkpointing**: For memory-efficient training

## Citation

If you use this code in your research, please cite:

```
@article{her2_pipeline_2024,
  title={Multi-task Deep Learning for HER2 Status Classification and Tumor Segmentation in Breast Cancer},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
