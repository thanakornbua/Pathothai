import os
import pydicom
import numpy as np
from PIL import Image
import openslide
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class Conversiontopng:
    @staticmethod
    def convert_dicom_svs_to_png(input_folder, output_folder, svs_level=0, max_workers=8, auto_downsample=False, max_images=None):
        """
        Batch convert DICOM (.dcm) and SVS (.svs) files to PNG with multithreading, progress bar, and optimization.
        
        Parameters:
            input_folder (str): Path to the folder containing DICOM and/or SVS files.
            output_folder (str): Path where PNG files will be saved.
            svs_level (int): SVS level to read (0 = full resolution). If auto_downsample=True, chooses optimal level.
            max_workers (int): Number of threads for parallel processing.
            auto_downsample (bool): Automatically choose lower SVS level for huge slides.
            max_images (int): Maximum number of images to process. If None, process all images.
        """
        os.makedirs(output_folder, exist_ok=True)

        def normalize_to_uint8(array):
            array = array.astype(float)
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
            return (array * 255).astype(np.uint8)

        def process_file(filename):
            filepath = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            output_path = os.path.join(output_folder, f"{name}.png")

            if os.path.exists(output_path):  # Skip already converted files
                return f"[SKIP] {filename}"

            try:
                start_time = time.time()
                if ext == '.dcm':  # Handle DICOM
                    ds = pydicom.dcmread(filepath)
                    pixel_array = ds.pixel_array
                    if pixel_array.ndim == 2:
                        img = Image.fromarray(normalize_to_uint8(pixel_array))
                    else:
                        img = Image.fromarray(pixel_array)
                    img.save(output_path)
                    elapsed_time = time.time() - start_time
                    return f"[DICOM] Saved: {filename} (Time: {elapsed_time:.2f}s)"

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
                    img.save(output_path)
                    elapsed_time = time.time() - start_time
                    return f"[SVS] Saved: {filename} (Time: {elapsed_time:.2f}s)"

                else:
                    return f"[SKIP: Unsupported] {filename}"

            except Exception as e:
                return f"[ERROR] {filename}: {e}"

        # Gather valid files
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.dcm', '.svs'))]
        tqdm.write(f"Found {len(files)} files to process.")

        # Limit files to process if max_images is set
        if max_images is not None:
            files = files[:max_images]

        # Process in parallel with progress bar
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Converting files", unit="file"):
                results.append(future.result())

        # Print summary
        tqdm.write("\n--- Conversion Summary ---")
        for r in results:
            tqdm.write(r)