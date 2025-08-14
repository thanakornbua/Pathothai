#!/usr/bin/env python3
"""
Debug script to test individual tile image creation and save to disk
This will help us verify if the tile data is correct by saving raw tiles
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from core.enhanced_image_loader import EnhancedLargeImageLoader

def test_single_tile_save():
    """Test saving a single tile to verify the image data"""
    
    app = QApplication(sys.argv)
    
    # Initialize loader
    loader = EnhancedLargeImageLoader()
    
    # Load SVS file
    svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    print(f"🔍 Testing individual tile from: {svs_file}")
    
    success = loader.load_image(svs_file)
    if not success:
        print("❌ Failed to load SVS file")
        return
    
    print("✅ SVS file loaded successfully")
    
    # Get a single tile
    tile_data = loader.get_tile(0, 1, 1, 256)  # Level 0, tile (1,1)
    
    if tile_data is None:
        print("❌ Failed to get tile data")
        return
    
    print(f"📊 Tile shape: {tile_data.shape}")
    print(f"📊 Tile dtype: {tile_data.dtype}")
    print(f"📊 Tile range: {tile_data.min()} - {tile_data.max()}")
    print(f"📊 Tile mean: {tile_data.mean():.1f}")
    
    # Save raw tile using PIL
    if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
        print("💾 Saving raw tile using PIL...")
        pil_img = Image.fromarray(tile_data, 'RGB')
        pil_img.save("debug_tile_raw.png")
        print("✅ Raw tile saved as debug_tile_raw.png")
        
        # Apply contrast enhancement if very bright
        if tile_data.mean() > 230:
            print("🔧 Applying contrast enhancement...")
            enhanced_data = tile_data.copy()
            tile_min = enhanced_data.min()
            tile_max = enhanced_data.max()
            if tile_max > tile_min:
                enhanced_data = ((enhanced_data.astype(np.float32) - tile_min) / 
                               (tile_max - tile_min) * 255).astype(np.uint8)
                print(f"✅ Enhanced range: {enhanced_data.min()} - {enhanced_data.max()}")
                print(f"✅ Enhanced mean: {enhanced_data.mean():.1f}")
                
                # Save enhanced tile
                pil_enhanced = Image.fromarray(enhanced_data, 'RGB')
                pil_enhanced.save("debug_tile_enhanced.png")
                print("✅ Enhanced tile saved as debug_tile_enhanced.png")
        
        # Test QImage creation
        print("🖼️  Testing QImage creation...")
        h, w, c = tile_data.shape
        bytes_per_line = 3 * w
        
        # Test with original data
        tile_bytes = tile_data.tobytes()
        q_image = QImage(tile_bytes, w, h, bytes_per_line, QImage.Format_RGB888)
        
        if not q_image.isNull():
            print("✅ QImage created successfully")
            
            # Test pixmap creation
            pixmap = QPixmap.fromImage(q_image)
            if not pixmap.isNull():
                print(f"✅ QPixmap created successfully: {pixmap.width()}x{pixmap.height()}")
                
                # Save QImage to check if conversion is working
                q_image.save("debug_tile_qimage.png")
                print("✅ QImage saved as debug_tile_qimage.png")
            else:
                print("❌ Failed to create QPixmap")
        else:
            print("❌ Failed to create QImage")
    else:
        print(f"❌ Unexpected tile shape: {tile_data.shape}")

if __name__ == "__main__":
    test_single_tile_save()
