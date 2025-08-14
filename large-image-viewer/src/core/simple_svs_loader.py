#!/usr/bin/env python3
"""
Simple, direct SVS file loader with built-in contrast enhancement.
No complex inheritance, just straightforward tile loading.
"""

import os
import time
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import openslide
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class SimpleSVSLoader:
    """Simple, direct SVS loader with built-in contrast enhancement."""
    
    def __init__(self):
        self.slide = None
        self.image_shape = None
        self.levels = 0
        self.level_dimensions = []
        self.level_downsamples = []
        self.tile_size = 256
        
    def load_svs_file(self, file_path: str) -> bool:
        """Load SVS file with OpenSlide."""
        try:
            print(f"🔍 Loading SVS file: {file_path}")
            
            # Open with OpenSlide
            self.slide = openslide.OpenSlide(file_path)
            
            # Get basic properties
            self.levels = self.slide.level_count
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            self.image_shape = (self.slide.dimensions[1], self.slide.dimensions[0])  # (height, width)
            
            print(f"✅ SVS loaded: {self.slide.dimensions[0]}x{self.slide.dimensions[1]}")
            print(f"📊 Levels: {self.levels}")
            for i, (w, h) in enumerate(self.level_dimensions):
                print(f"   Level {i}: {w}x{h}, downsample: {self.level_downsamples[i]:.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SVS: {e}")
            return False
    
    def get_tile_with_enhancement(self, level: int, tile_x: int, tile_y: int) -> Optional[QPixmap]:
        """Get a tile with automatic contrast enhancement for SVS files."""
        if not self.slide:
            return None
            
        try:
            # Calculate tile position
            x = tile_x * self.tile_size
            y = tile_y * self.tile_size
            
            # Get level dimensions
            if level >= len(self.level_dimensions):
                return None
                
            level_w, level_h = self.level_dimensions[level]
            
            # Clamp tile to image bounds
            width = min(self.tile_size, level_w - x)
            height = min(self.tile_size, level_h - y)
            
            if width <= 0 or height <= 0:
                return None
            
            # Read tile from OpenSlide
            pil_tile = self.slide.read_region((x, y), level, (width, height))
            
            # Convert RGBA to RGB with white background
            if pil_tile.mode == 'RGBA':
                rgb_tile = Image.new('RGB', pil_tile.size, (255, 255, 255))
                rgb_tile.paste(pil_tile, mask=pil_tile.split()[3])
                pil_tile = rgb_tile
            elif pil_tile.mode != 'RGB':
                pil_tile = pil_tile.convert('RGB')
            
            # Convert to numpy array
            tile_array = np.array(pil_tile, dtype=np.uint8)
            
            # Apply contrast enhancement for bright SVS tiles
            tile_mean = tile_array.mean()
            if tile_mean > 230:
                print(f"🔧 Enhancing bright tile (mean={tile_mean:.1f}) at ({tile_x},{tile_y})")
                
                # Apply aggressive contrast stretching
                tile_min = tile_array.min()
                tile_max = tile_array.max()
                
                if tile_max > tile_min:
                    # Stretch to full 0-255 range
                    enhanced = ((tile_array.astype(np.float32) - tile_min) / 
                              (tile_max - tile_min) * 255.0).astype(np.uint8)
                    
                    # Verify enhancement
                    new_mean = enhanced.mean()
                    center_y, center_x = enhanced.shape[0]//2, enhanced.shape[1]//2
                    center_pixel = enhanced[center_y, center_x]
                    
                    print(f"   ✅ Enhanced: {tile_min}-{tile_max} → 0-255, mean {tile_mean:.1f} → {new_mean:.1f}")
                    print(f"   🎨 Center pixel: R={center_pixel[0]}, G={center_pixel[1]}, B={center_pixel[2]}")
                    
                    tile_array = enhanced
            
            # Create QImage directly from enhanced array
            h, w = tile_array.shape[:2]
            bytes_per_line = 3 * w
            
            # Ensure contiguous array for QImage
            if not tile_array.flags['C_CONTIGUOUS']:
                tile_array = np.ascontiguousarray(tile_array)
            
            # Create QImage
            q_image = QImage(tile_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            if q_image.isNull():
                print(f"❌ Failed to create QImage for tile ({tile_x},{tile_y})")
                return None
            
            # Create QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            if pixmap.isNull():
                print(f"❌ Failed to create QPixmap for tile ({tile_x},{tile_y})")
                return None
            
            # Verify pixmap content
            test_img = pixmap.toImage()
            if not test_img.isNull() and test_img.width() > 10 and test_img.height() > 10:
                center_color = test_img.pixelColor(test_img.width()//2, test_img.height()//2)
                print(f"   ✅ Pixmap created: {pixmap.width()}x{pixmap.height()}")
                print(f"   🔍 Pixmap center color: R={center_color.red()}, G={center_color.green()}, B={center_color.blue()}")
            
            return pixmap
            
        except Exception as e:
            print(f"❌ Error getting tile ({tile_x},{tile_y}): {e}")
            return None
    
    def get_level_dimensions(self, level: int) -> Optional[Tuple[int, int]]:
        """Get dimensions for a specific level."""
        if not self.slide or level >= len(self.level_dimensions):
            return None
        return self.level_dimensions[level]
    
    def get_level_count(self) -> int:
        """Get number of levels."""
        return self.levels if self.slide else 0
    
    def cleanup(self):
        """Clean up resources."""
        if self.slide:
            self.slide.close()
            self.slide = None
