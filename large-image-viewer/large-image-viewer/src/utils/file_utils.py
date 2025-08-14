def is_image_file(filepath):
    return filepath.lower().endswith(('.png', '.jpg', '.jpeg'))

def get_file_size(filepath):
    import os
    return os.path.getsize(filepath)

def validate_image_file(filepath):
    if not is_image_file(filepath):
        raise ValueError("File is not a valid image format. Supported formats: PNG, JPG, JPEG.")
    
    file_size = get_file_size(filepath)
    if file_size > 1 * 1024 * 1024 * 1024:  # 1GB in bytes
        raise ValueError("File size exceeds 1GB limit.")
    
    return True

def get_file_extension(filepath):
    import os
    return os.path.splitext(filepath)[1]