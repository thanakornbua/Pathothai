def extract_channels(image):
    """Extract RGB channels from the image."""
    r, g, b = image.split()
    return r, g, b

def merge_channels(r, g, b):
    """Merge RGB channels back into a single image."""
    return Image.merge("RGB", (r, g, b))

def adjust_brightness(image, factor):
    """Adjust the brightness of the image."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_contrast(image, factor):
    """Adjust the contrast of the image."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def apply_filter(image, filter_type):
    """Apply a filter to the image based on the filter type."""
    if filter_type == 'BLUR':
        return image.filter(ImageFilter.BLUR)
    elif filter_type == 'CONTOUR':
        return image.filter(ImageFilter.CONTOUR)
    elif filter_type == 'DETAIL':
        return image.filter(ImageFilter.DETAIL)
    else:
        return image

def resize_image(image, new_size):
    """Resize the image to the new size."""
    return image.resize(new_size, Image.ANTIALIAS)