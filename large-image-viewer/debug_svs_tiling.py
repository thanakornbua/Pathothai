#!/usr/bin/env python3
"""
Debug script for SVS tiling issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.core.enhanced_image_loader import EnhancedLargeImageLoader

def debug_svs_tiling():
    """Debug SVS tiling functionality."""
    
    # Test with the SVS file mentioned in the error
    svs_file = r"C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    
    if not os.path.exists(svs_file):
        print(f"SVS file not found: {svs_file}")
        return
    
    print("="*60)
    print("DEBUG: SVS Tiling Test")
    print("="*60)
    
    # Load the WSI
    loader = EnhancedLargeImageLoader()
    success = loader.load_image(svs_file)
    
    if not success:
        print("❌ Failed to load SVS file")
        return
    
    print(f"✅ SVS file loaded successfully")
    print(f"📐 Image shape: {loader.image_shape}")
    print(f"🎨 Channels: {loader.channels}")
    print(f"📊 Zoom levels: {len(loader.zoom_levels)}")
    
    for i, level in enumerate(loader.zoom_levels):
        print(f"   Level {i}: {level['width']}x{level['height']}, downsample: {level['downsample']:.2f}")
    
    print("\n🧪 Testing tile extraction:")
    
    # Test tile extraction at different levels
    test_tiles = [
        (0, 0, 0),  # Level 0, tile (0,0)
        (0, 1, 0),  # Level 0, tile (1,0)
        (0, 0, 1),  # Level 0, tile (0,1)
        (1, 0, 0),  # Level 1, tile (0,0)
        (2, 0, 0),  # Level 2, tile (0,0)
    ]
    
    for level, tile_x, tile_y in test_tiles:
        if level < len(loader.zoom_levels):
            tile_data = loader.get_tile(level, tile_x, tile_y)
            if tile_data is not None:
                print(f"✅ Level {level}, Tile ({tile_x},{tile_y}): {tile_data.shape}, dtype: {tile_data.dtype}")
                print(f"   Data range: {tile_data.min()} - {tile_data.max()}")
                
                # Check if the tile has actual content (not all zeros or single value)
                unique_values = len(np.unique(tile_data))
                if unique_values < 10:
                    print(f"   ⚠️  Warning: Tile has only {unique_values} unique values - might be empty")
                else:
                    print(f"   ✅ Tile has {unique_values} unique values - looks good")
            else:
                print(f"❌ Level {level}, Tile ({tile_x},{tile_y}): No data returned")
        else:
            print(f"❌ Level {level} does not exist")
    
    print(f"\n🧹 Cleanup")
    loader.cleanup()
    print("Debug test completed")

if __name__ == "__main__":
    debug_svs_tiling()
