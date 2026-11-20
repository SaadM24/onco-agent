"""Deterministic synthetic corruptions for research stress tests, not an external set."""
from PIL import Image, ImageEnhance, ImageFilter


def perturb(image: Image.Image, name: str):
    if name == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=1.5))
    if name == 'dark':
        return ImageEnhance.Brightness(image).enhance(.65)
    if name == 'bright':
        return ImageEnhance.Brightness(image).enhance(1.35)
    if name == 'color':
        return ImageEnhance.Color(image).enhance(1.5)
    raise ValueError(f'Unknown synthetic perturbation {name}')
