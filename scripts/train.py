import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import openslide
import pydicom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class WholeSlideDataset(Dataset):
    """Dataset for handling whole slide images with patch extraction"""
    
    def __init__(self, slide_paths, labels, patch_size=224, patches_per_slide=50, 
                 transform=None, svs_level=1, cache_patches=True, annotation_paths=None):
        self.slide_paths = slide_paths
        self.labels = labels
        self.patch_size = patch_size
        self.patches_per_slide = patches_per_slide
        self.transform = transform
        self.svs_level = svs_level
        self.cache_patches = cache_patches
        self.patch_cache = {}
        self.annotation_paths = annotation_paths if annotation_paths is not None else [None] * len(slide_paths)
        self.regions = [self._parse_annotation(xml) if xml else None for xml in self.annotation_paths]

    def _parse_annotation(self, xml_path):
        """Parse XML annotation file and return list of regions with label and coordinates."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            regions = []
            # For each Annotation, get its Name and all Regions under it
            for annotation in root.findall('.//Annotation'):
                annotation_label = annotation.attrib.get('Name', None)
                for region in annotation.findall('.//Region'):
                    label = region.attrib.get('Text', None)
                    if not label:
                        label = annotation_label
                    coords = []
                    for vertex in region.findall('.//Vertex'):
                        x = float(vertex.attrib['X'])
                        y = float(vertex.attrib['Y'])
                        coords.append((x, y))
                    regions.append({'label': label, 'coords': coords})
            return regions
        except Exception as e:
            print(f"Error parsing annotation {xml_path}: {e}")
            return []
        
    def __len__(self):
        return len(self.slide_paths) * self.patches_per_slide
    
    def __getitem__(self, idx):
        slide_idx = idx // self.patches_per_slide
        patch_idx = idx % self.patches_per_slide
        
        slide_path = self.slide_paths[slide_idx]
        label = self.labels[slide_idx]
        
        # Check cache first
        cache_key = f"{slide_path}_{patch_idx}"
        if self.cache_patches and cache_key in self.patch_cache:
            patch = self.patch_cache[cache_key]
        else:
            patch = self._extract_patch(slide_path, patch_idx)
            if self.cache_patches:
                self.patch_cache[cache_key] = patch
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label
    
    def _extract_patch(self, slide_path, patch_idx):
        """Extract a patch from the slide"""
        try:
            if slide_path.lower().endswith('.svs'):
                return self._extract_svs_patch(slide_path, patch_idx)
            elif slide_path.lower().endswith('.dcm'):
                return self._extract_dicom_patch(slide_path, patch_idx)
            else:
                raise ValueError(f"Unsupported file format: {slide_path}")
        except Exception as e:
            print(f"Error extracting patch from {slide_path}: {e}")
            # Return a blank patch as fallback
            return Image.new('RGB', (self.patch_size, self.patch_size), color='white')
    
    def _extract_svs_patch(self, slide_path, patch_idx):
        """Extract patch from SVS file, using annotation if available."""
        slide = openslide.OpenSlide(slide_path)
        level_dims = slide.level_dimensions[self.svs_level]
        slide_idx = self.slide_paths.index(slide_path)
        regions = self.regions[slide_idx] if self.regions else None
        random.seed(hash(f"{slide_path}_{patch_idx}") % (2**32))
        if regions and len(regions) > 0:
            # Pick a random region, then a random point inside its bounding box
            region = random.choice(regions)
            xs = [pt[0] for pt in region['coords']]
            ys = [pt[1] for pt in region['coords']]
            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))
            max_x = max(min_x, max_x - self.patch_size)
            max_y = max(min_y, max_y - self.patch_size)
            if max_x > min_x and max_y > min_y:
                x = random.randint(min_x, max_x)
                y = random.randint(min_y, max_y)
            else:
                x = random.randint(0, level_dims[0] - self.patch_size)
                y = random.randint(0, level_dims[1] - self.patch_size)
        else:
            # Fallback: random patch anywhere
            x = random.randint(0, max(0, level_dims[0] - self.patch_size))
            y = random.randint(0, max(0, level_dims[1] - self.patch_size))
        patch = slide.read_region((x, y), self.svs_level, 
                                 (self.patch_size, self.patch_size)).convert('RGB')
        slide.close()
        return patch
    
    def _extract_dicom_patch(self, slide_path, patch_idx):
        """Extract patch from DICOM file"""
        ds = pydicom.dcmread(slide_path)
        pixel_array = ds.pixel_array
        
        if pixel_array.ndim == 2:
            # Convert grayscale to RGB
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        h, w = pixel_array.shape[:2]
        max_x = max(0, w - self.patch_size)
        max_y = max(0, h - self.patch_size)
        
        random.seed(hash(f"{slide_path}_{patch_idx}") % (2**32))
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        patch = pixel_array[y:y+self.patch_size, x:x+self.patch_size]
        patch = Image.fromarray(patch.astype(np.uint8))
        return patch

class WSITrainer:
    """Whole Slide Image Trainer with patch-based learning"""
    
    def __init__(self, model_name='resnet50', num_classes=2, device=None, 
                 checkpoint_dir='checkpoints', mixed_precision=True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.mixed_precision = mixed_precision
        # Initialize model
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "runs"))
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = []
        
    def _create_model(self):
        """Create and initialize the model"""
        if self.model_name.lower() == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name.lower() == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    def prepare_data(self, data_dir, patch_size=224, patches_per_slide=50, 
                    batch_size=32, val_split=0.2, num_workers=4):
        """Prepare datasets and dataloaders"""
        
        # Collect slide files and labels
        slide_paths = []
        labels = []
        class_names = []
        
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.svs', '.dcm')):
                        slide_paths.append(os.path.join(class_dir, file))
                        labels.append(class_idx)
        
        print(f"Found {len(slide_paths)} slides across {len(class_names)} classes")
        
        # Split into train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            slide_paths, labels, test_size=val_split, random_state=42, stratify=labels)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = WholeSlideDataset(
            train_paths, train_labels, patch_size, patches_per_slide, train_transform)
        val_dataset = WholeSlideDataset(
            val_paths, val_labels, patch_size, patches_per_slide, val_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True)
        
        self.class_names = class_names
        return len(train_dataset), len(val_dataset)
    
    def setup_training(self, learning_rate=0.001, weight_decay=1e-4):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved with validation accuracy: {val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            self.training_history = checkpoint['training_history']
            print(f"Resumed from epoch {self.current_epoch}")
            return True
        return False
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        try:
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{acc:.2f}%'
                })
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            raise
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs, save_every=5):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_epoch = self.current_epoch
        
        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch
                # Train
                train_loss, train_acc = self.train_epoch()
                # Validate
                val_loss, val_acc = self.validate()
                # Update learning rate
                self.scheduler.step(val_acc)
                # Save training history
                epoch_info = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.training_history.append(epoch_info)
                # TensorBoard logging
                try:
                    self.writer.add_scalar("train/loss", train_loss, epoch)
                    self.writer.add_scalar("train/accuracy", train_acc, epoch)
                    self.writer.add_scalar("val/loss", val_loss, epoch)
                    self.writer.add_scalar("val/accuracy", val_acc, epoch)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], epoch)
                except Exception:
                    pass
                # Print results
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 50)
                # Save checkpoint
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                if (epoch + 1) % save_every == 0 or is_best:
                    self.save_checkpoint(epoch, val_acc, is_best)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(self.current_epoch, val_acc)
            try:
                self.writer.close()
            except Exception:
                pass
            raise
        try:
            self.writer.close()
        except Exception:
            pass
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

# Utility functions for easy notebook usage
def create_trainer(model_name='resnet50', num_classes=2, checkpoint_dir='checkpoints'):
    """Create a WSI trainer instance"""
    return WSITrainer(
        model_name=model_name,
        num_classes=num_classes,
        checkpoint_dir=checkpoint_dir,
        mixed_precision=True
    )

def quick_train(data_dir, model_name='resnet50', num_classes=2, num_epochs=10, 
                patch_size=224, patches_per_slide=50, batch_size=16):
    """Quick training function for notebook use"""
    
    print(f"🚀 Starting quick training with {model_name}")
    
    # Create trainer
    trainer = create_trainer(model_name, num_classes)
    
    # Prepare data
    train_size, val_size = trainer.prepare_data(
        data_dir=data_dir,
        patch_size=patch_size,
        patches_per_slide=patches_per_slide,
        batch_size=batch_size,
        num_workers=2
    )
    
    # Setup training
    trainer.setup_training(learning_rate=0.001)
    
    # Try to load checkpoint
    trainer.load_checkpoint()
    
    # Train
    trainer.train(num_epochs)
    
    return trainer

def main():
    # Configuration
    config = {
        'data_dir': 'data/train',  # Directory with class subdirectories containing .svs/.dcm files
        'model_name': 'resnet50',  # or 'efficientnet_b0'
        'num_classes': 2,  # HER2 positive/negative
        'patch_size': 224,
        'patches_per_slide': 100,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'val_split': 0.2,
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'mixed_precision': True
    }
    
    # Create trainer
    trainer = WSITrainer(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        checkpoint_dir=config['checkpoint_dir'],
        mixed_precision=config['mixed_precision']
    )
    
    # Prepare data
    train_size, val_size = trainer.prepare_data(
        data_dir=config['data_dir'],
        patch_size=config['patch_size'],
        patches_per_slide=config['patches_per_slide'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        num_workers=config['num_workers']
    )
    
    # Setup training
    trainer.setup_training(
        learning_rate=config['learning_rate']
    )
    
    # Try to resume from checkpoint
    trainer.load_checkpoint()
    
    # Start training
    trainer.train(config['num_epochs'])

if __name__ == "__main__":
    main()