#!/usr/bin/env python3
"""
Analyze SVS tile content to find tissue areas vs background areas.
"""

import sys
import os
import numpy as np

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from core.enhanced_image_loader import EnhancedLargeImageLoader

def analyze_svs_tissue_distribution():
    """Analyze SVS file to find where tissue content is located."""
    
    # Find an SVS file
    svs_file = "C:/Users/tanth/Desktop/Pathothai/data/Yale_HER2_cohort/SVS_positive/Her2Pos_Case_47.svs"
    
    if not os.path.exists(svs_file):
        print(f"SVS file not found: {svs_file}")
        return
    
    print("="*60)
    print("DEBUG: SVS Tissue Distribution Analysis")
    print("="*60)
    
    # Load the WSI
    loader = EnhancedLargeImageLoader()
    success = loader.load_image(svs_file)
    
    if not success:
        print("❌ Failed to load SVS file")
        return
    
    print(f"✅ SVS file loaded successfully")
    print(f"📐 Image shape: {loader.image_shape}")
    print(f"📊 Zoom levels: {len(loader.zoom_levels)}")
    
    # Analyze multiple tiles across the image to find tissue
    level = 2  # Use lower resolution level for faster analysis
    level_info = loader.zoom_levels[level]
    level_width = level_info['width']
    level_height = level_info['height']
    tile_size = 256
    
    print(f"\n🔍 Analyzing tiles at level {level} ({level_width}x{level_height}):")
    
    # Sample tiles from different regions
    max_tiles_x = level_width // tile_size
    max_tiles_y = level_height // tile_size
    
    print(f"Max tiles: {max_tiles_x}x{max_tiles_y}")
    
    # Test tiles from different regions
    test_positions = [
        (0, 0),  # Top-left (likely background)
        (max_tiles_x//4, max_tiles_y//4),  # Quarter
        (max_tiles_x//2, max_tiles_y//2),  # Center
        (3*max_tiles_x//4, max_tiles_y//4),  # Three-quarter
        (max_tiles_x//4, 3*max_tiles_y//4),  # Lower quarter
        (3*max_tiles_x//4, 3*max_tiles_y//4),  # Lower three-quarter
        (max_tiles_x-1, max_tiles_y-1),  # Bottom-right
    ]
    
    tissue_tiles = []
    background_tiles = []
    
    for tile_x, tile_y in test_positions:
        if tile_x >= max_tiles_x or tile_y >= max_tiles_y:
            continue
            
        tile_data = loader.get_tile(level, tile_x, tile_y)
        if tile_data is not None:
            mean_val = tile_data.mean()
            std_val = tile_data.std()
            min_val = tile_data.min()
            max_val = tile_data.max()
            
            # Determine if this looks like tissue or background
            # Background typically has high mean values (bright) and low std deviation
            # Tissue typically has lower mean values and higher variation
            is_background = mean_val > 230 and std_val < 5
            is_tissue = mean_val < 200 or std_val > 10
            
            category = "🔬 TISSUE" if is_tissue else ("📄 BACKGROUND" if is_background else "❓ UNKNOWN")
            
            print(f"  Tile ({tile_x:2d},{tile_y:2d}): mean={mean_val:5.1f}, std={std_val:4.1f}, min={min_val:3d}, max={max_val:3d} - {category}")
            
            if is_tissue:
                tissue_tiles.append((tile_x, tile_y, mean_val, std_val))
            elif is_background:
                background_tiles.append((tile_x, tile_y, mean_val, std_val))
    
    print(f"\n📊 Summary:")
    print(f"  🔬 Tissue tiles found: {len(tissue_tiles)}")
    print(f"  📄 Background tiles found: {len(background_tiles)}")
    
    if len(tissue_tiles) > 0:
        print(f"\n🔬 Best tissue tiles to test:")
        # Sort by standard deviation (more variation = more likely to be tissue)
        tissue_tiles.sort(key=lambda x: x[3], reverse=True)
        for i, (tx, ty, mean_val, std_val) in enumerate(tissue_tiles[:3]):
            print(f"  {i+1}. Tile ({tx},{ty}): mean={mean_val:.1f}, std={std_val:.1f}")
    else:
        print(f"\n⚠️  No obvious tissue tiles found in sampled areas.")
        print(f"    The SVS file may contain mostly background, or tissue may be")
        print(f"    in different areas. Try looking at the thumbnail or adjust")
        print(f"    the viewport to find tissue regions.")
    
    # Also check the thumbnail to see if it shows tissue
    if hasattr(loader, 'thumbnail') and loader.thumbnail is not None:
        thumb = loader.thumbnail
        thumb_mean = thumb.mean()
        thumb_std = thumb.std()
        print(f"\n🖼️  Thumbnail analysis:")
        print(f"   Mean: {thumb_mean:.1f}, Std: {thumb_std:.1f}")
        if thumb_mean > 230:
            print(f"   ⚠️  Thumbnail also appears very bright - mostly background")
        else:
            print(f"   ✅ Thumbnail shows darker regions - tissue likely present")

if __name__ == "__main__":
    analyze_svs_tissue_distribution()
