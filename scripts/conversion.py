import os
import pydicom
import numpy as np
from PIL import Image
import openslide
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
import torch
import torchvision.transforms as transforms
from multiprocessing import cpu_count
import gc
class Conversiontopng:
    def __init__(self, use_cuda=True, cuda_batch_size=32):
        """
        Initialize the converter with CUDA optimization options.
        
        Parameters:
            use_cuda (bool): Whether to use CUDA for acceleration
            cuda_batch_size (int): Batch size for CUDA processing
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.cuda_batch_size = cuda_batch_size
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        if self.use_cuda:
            print(f"🚀 CUDA acceleration enabled on {torch.cuda.get_device_name(0)}")
            print(f"💾 CUDA memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("💻 Using CPU processing")
    
    def _optimize_workers(self, max_workers=None):
        """Automatically determine optimal number of workers"""
        if max_workers is not None:
            return min(max_workers, cpu_count())
        
        # Calculate optimal workers based on system resources
        cpu_cores = cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative estimate: 1 worker per 2GB RAM, max 2x CPU cores
        optimal_workers = min(int(memory_gb / 2), cpu_cores * 2, 16)
        return max(1, optimal_workers)
    
    def _cuda_normalize_batch(self, batch_arrays):
        """Normalize a batch of arrays using CUDA"""
        if not self.use_cuda or not batch_arrays:
            return [self._normalize_to_uint8_cpu(arr) for arr in batch_arrays]
        
        try:
            # Convert to torch tensors and move to GPU
            tensors = []
            for arr in batch_arrays:
                if arr.ndim == 2:
                    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(arr.astype(np.float32))
                tensors.append(tensor.to(self.device))
            
            # Batch normalize on GPU
            normalized = []
            for tensor in tensors:
                # Normalize to 0-1 range
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
                # Scale to 0-255 and convert to uint8
                tensor_uint8 = (tensor_norm * 255).byte()
                normalized.append(tensor_uint8.cpu().numpy())
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            return normalized
            
        except Exception as e:
            print(f"⚠️ CUDA normalization failed: {e}, falling back to CPU")
            return [self._normalize_to_uint8_cpu(arr) for arr in batch_arrays]
    
    def _normalize_to_uint8_cpu(self, array):
        """CPU-based normalization (fallback)"""
        array = array.astype(float)
        array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        return (array * 255).astype(np.uint8)
    
    @staticmethod
    def convert_dicom_svs_to_png(input_folder, output_folder, svs_level=0, max_workers=None, 
                                auto_downsample=True, max_images=None, use_cuda=True, 
                                cuda_batch_size=32, progress_callback=None):
        """
        Batch convert DICOM (.dcm) and SVS (.svs) files to PNG with CUDA acceleration and optimized multithreading.

        Parameters:
            input_folder (str): Path to the folder containing DICOM and/or SVS files.
            output_folder (str): Path where PNG files will be saved.
            svs_level (int): SVS level to read (0 = full resolution). If auto_downsample=True, chooses optimal level.
            max_workers (int or None): Number of threads for parallel processing. Auto-optimized if None.
            auto_downsample (bool): Automatically choose lower SVS level for huge slides.
            max_images (int or None): Maximum number of images to process. If None, process all.
            use_cuda (bool): Whether to use CUDA acceleration for image processing.
            cuda_batch_size (int): Batch size for CUDA processing.
            progress_callback (callable): Optional callback function for progress updates.
        
        Returns:
            dict: Statistics about the conversion process
        """
        # Create converter instance
        converter = Conversiontopng(use_cuda=use_cuda, cuda_batch_size=cuda_batch_size)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Optimize worker count
        optimal_workers = converter._optimize_workers(max_workers)
        print(f"🔧 Using {optimal_workers} worker threads")
        
        # Statistics tracking
        stats = {
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': time.time(),
            'processing_times': []
        }

        def process_file_optimized(filename):
            """Optimized file processing function"""
            filepath = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            output_path = os.path.join(output_folder, f"{name}.png")
            
            start_time = time.time()

            if os.path.exists(output_path):  # Skip already converted files
                return {"status": "skip", "filename": filename, "time": 0}

            try:
                if ext == '.dcm':  # Handle DICOM
                    ds = pydicom.dcmread(filepath)
                    pixel_array = ds.pixel_array
                    
                    # Use CUDA for normalization if available
                    if pixel_array.ndim == 2:
                        normalized = converter._cuda_normalize_batch([pixel_array])[0]
                        img = Image.fromarray(normalized)
                    else:
                        img = Image.fromarray(pixel_array)
                    
                    # Optimize PNG saving
                    img.save(output_path, optimize=True, compress_level=6)
                    processing_time = time.time() - start_time
                    
                    return {
                        "status": "success", 
                        "filename": filename, 
                        "type": "DICOM",
                        "time": processing_time,
                        "size": os.path.getsize(output_path)
                    }

                elif ext == '.svs':  # Handle SVS
                    slide = openslide.OpenSlide(filepath)
                    level_dims = slide.level_dimensions

                    # Auto-select optimal level if enabled
                    level_to_use = svs_level
                    if auto_downsample:
                        for i, dims in enumerate(level_dims):
                            if max(dims) < 4000:  # pick level with width/height < 4000px
                                level_to_use = i
                                break

                    img = slide.read_region((0, 0), level_to_use, level_dims[level_to_use]).convert("RGB")
                    
                    # Optimize PNG saving
                    img.save(output_path, optimize=True, compress_level=6)
                    slide.close()
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "status": "success", 
                        "filename": filename, 
                        "type": "SVS",
                        "level": level_to_use,
                        "dimensions": level_dims[level_to_use],
                        "time": processing_time,
                        "size": os.path.getsize(output_path)
                    }

                else:
                    return {"status": "unsupported", "filename": filename, "time": 0}

            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    "status": "error", 
                    "filename": filename, 
                    "error": str(e),
                    "time": processing_time
                }
            finally:
                # Force garbage collection to free memory
                gc.collect()

        # Gather valid files
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.dcm', '.svs'))]
        if max_images is not None:
            files = files[:max_images]
        
        stats['total_files'] = len(files)
        tqdm.write(f"📁 Found {len(files)} files to process")
        
        if len(files) == 0:
            tqdm.write("⚠️ No valid files found!")
            return stats

        # Process files with optimized multithreading
        results = []
        try:
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                futures = {executor.submit(process_file_optimized, f): f for f in files}
                
                progress_bar = tqdm(as_completed(futures), total=len(futures), 
                                  desc="🔄 Converting", unit="file")
                
                for future in progress_bar:
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update statistics
                        if result["status"] == "success":
                            stats['processed'] += 1
                            stats['processing_times'].append(result['time'])
                        elif result["status"] == "skip":
                            stats['skipped'] += 1
                        else:
                            stats['errors'] += 1
                        
                        # Update progress bar with current stats
                        progress_bar.set_postfix({
                            'Processed': stats['processed'],
                            'Skipped': stats['skipped'],
                            'Errors': stats['errors']
                        })
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(stats, result)
                            
                    except Exception as e:
                        stats['errors'] += 1
                        results.append({
                            "status": "error",
                            "filename": "unknown",
                            "error": str(e),
                            "time": 0
                        })

        except KeyboardInterrupt:
            tqdm.write("\n⏹️ Conversion interrupted by user")
            stats['interrupted'] = True
        
        # Calculate final statistics
        total_time = time.time() - stats['start_time']
        stats['total_time'] = total_time
        if stats['processing_times']:
            stats['avg_time_per_file'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        
        # Print detailed summary
        Conversiontopng._print_conversion_summary(stats, results)
        
        return stats
    
    @staticmethod
    def _print_conversion_summary(stats, results):
        """Print detailed conversion summary"""
        tqdm.write("\n" + "="*60)
        tqdm.write("🎯 CONVERSION SUMMARY")
        tqdm.write("="*60)
        
        tqdm.write(f"📊 Total files found: {stats['total_files']}")
        tqdm.write(f"✅ Successfully processed: {stats['processed']}")
        tqdm.write(f"⏭️ Skipped (already exist): {stats['skipped']}")
        tqdm.write(f"❌ Errors: {stats['errors']}")
        tqdm.write(f"⏱️ Total time: {stats['total_time']:.2f} seconds")
        
        if stats.get('avg_time_per_file'):
            tqdm.write(f"📈 Average time per file: {stats['avg_time_per_file']:.2f} seconds")
            tqdm.write(f"🚀 Processing speed: {stats['processed']/stats['total_time']:.2f} files/second")
        
        # Print file type breakdown
        success_results = [r for r in results if r['status'] == 'success']
        if success_results:
            dicom_count = len([r for r in success_results if r.get('type') == 'DICOM'])
            svs_count = len([r for r in success_results if r.get('type') == 'SVS'])
            
            tqdm.write(f"\n📋 File type breakdown:")
            tqdm.write(f"   🩻 DICOM files: {dicom_count}")
            tqdm.write(f"   🔬 SVS files: {svs_count}")
            
            # Calculate total output size
            total_size = sum(r.get('size', 0) for r in success_results)
            tqdm.write(f"💾 Total output size: {total_size / 1024**2:.1f} MB")
        
        # Print errors if any
        error_results = [r for r in results if r['status'] == 'error']
        if error_results:
            tqdm.write(f"\n❌ Error details:")
            for error in error_results[:5]:  # Show first 5 errors
                tqdm.write(f"   {error['filename']}: {error['error']}")
            if len(error_results) > 5:
                tqdm.write(f"   ... and {len(error_results) - 5} more errors")

# Convenience functions for easy notebook usage
def quick_convert(input_folder, output_folder, max_images=None, use_cuda=True):
    """
    Quick conversion function with optimal settings
    
    Parameters:
        input_folder (str): Input directory path
        output_folder (str): Output directory path  
        max_images (int): Maximum number of images to process
        use_cuda (bool): Whether to use CUDA acceleration
    
    Returns:
        dict: Conversion statistics
    """
    return Conversiontopng.convert_dicom_svs_to_png(
        input_folder=input_folder,
        output_folder=output_folder,
        max_images=max_images,
        use_cuda=use_cuda,
        auto_downsample=True,
        max_workers=None  # Auto-optimize
    )

def batch_convert_folders(folder_list, output_base, **kwargs):
    """
    Convert multiple folders in batch
    
    Parameters:
        folder_list (list): List of input folder paths
        output_base (str): Base output directory
        **kwargs: Additional arguments for convert_dicom_svs_to_png
    
    Returns:
        dict: Combined statistics from all conversions
    """
    combined_stats = {
        'total_files': 0,
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'folders_processed': 0
    }
    
    for folder in folder_list:
        folder_name = os.path.basename(folder)
        output_folder = os.path.join(output_base, folder_name)
        
        print(f"\n🗂️ Processing folder: {folder_name}")
        stats = Conversiontopng.convert_dicom_svs_to_png(
            input_folder=folder,
            output_folder=output_folder,
            **kwargs
        )
        
        # Accumulate statistics
        for key in ['total_files', 'processed', 'skipped', 'errors']:
            combined_stats[key] += stats.get(key, 0)
        combined_stats['folders_processed'] += 1
    
    print(f"\n🎉 Batch conversion complete!")
    print(f"📁 Folders processed: {combined_stats['folders_processed']}")
    print(f"📊 Total files: {combined_stats['total_files']}")
    print(f"✅ Total processed: {combined_stats['processed']}")
    
    return combined_stats
