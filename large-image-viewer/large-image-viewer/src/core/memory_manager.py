class MemoryManager:
    def __init__(self):
        self.memory_limit = self.get_memory_limit()
        self.loaded_images = {}

    def get_memory_limit(self):
        import psutil
        return psutil.virtual_memory().available

    def load_image(self, image_path):
        import os
        if os.path.exists(image_path):
            image_size = os.path.getsize(image_path)
            if image_size > self.memory_limit:
                raise MemoryError("Image size exceeds available memory.")
            # Load the image into memory (placeholder for actual loading logic)
            self.loaded_images[image_path] = None  # Replace None with actual image data
            return self.loaded_images[image_path]
        else:
            raise FileNotFoundError("Image file not found.")

    def unload_image(self, image_path):
        if image_path in self.loaded_images:
            del self.loaded_images[image_path]

    def clear_memory(self):
        self.loaded_images.clear()