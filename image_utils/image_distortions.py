# non differentiable image distortions for testing

from PIL import Image, ImageFilter
import random
import numpy as np
import cv2
import torchvision.transforms as transforms

class RandomDistortion():
    def __init__(self, 
                 jpeg_compression=False,
                 add_gaussian_noise=False,
                 gaussian_blur=False,
                 median_blur=False,
                 salt_and_pepper_noise=False,
                 resize_then_recover=False,
                 brightness_transform=False,
                 random_crop_then_recover=False,
                 random_translate=False,
                 random_rotate=False,
                 return_distortion_type=False):
        self.return_distortion_type = return_distortion_type
        self.distortions = []
        if jpeg_compression: self.distortions.append('jpeg_compression')
        if add_gaussian_noise: self.distortions.append('add_gaussian_noise')
        if gaussian_blur: self.distortions.append('gaussian_blur')
        if median_blur: self.distortions.append('median_blur')
        if salt_and_pepper_noise: self.distortions.append('salt_and_pepper_noise')
        if resize_then_recover: self.distortions.append('resize_then_recover')
        if brightness_transform: self.distortions.append('brightness_transform')
        if random_crop_then_recover: self.distortions.append('random_crop_then_recover')
        if random_translate: self.distortions.append('random_translate')
        if random_rotate: self.distortions.append('random_rotate')

    def __call__(self, img):
        distoration = random.choice(self.distortions)
        if self.return_distortion_type:
            if distoration == 'jpeg_compression':
                img, strength = jpeg_compression(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'add_gaussian_noise':
                img, strength = add_gaussian_noise(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'gaussian_blur':
                img, strength = gaussian_blur(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'median_blur':
                img, strength = median_blur(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'salt_and_pepper_noise':
                img, strength = salt_and_pepper_noise(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'resize_then_recover':
                img, strength = resize_then_recover(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'brightness_transform':
                img, strength = brightness_transform(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_crop_then_recover':
                img, strength = random_crop_then_recover(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_translate':
                img, strength = random_translate(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_rotate':
                img, strength = random_rotate(img, return_distortion_type=self.return_distortion_type)
            return img, distoration, strength
        else:
            if distoration == 'jpeg_compression':
                img = jpeg_compression(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'add_gaussian_noise':
                img = add_gaussian_noise(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'gaussian_blur':
                img = gaussian_blur(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'median_blur':
                img = median_blur(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'salt_and_pepper_noise':
                img = salt_and_pepper_noise(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'resize_then_recover':
                img = resize_then_recover(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'brightness_transform':
                img = brightness_transform(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_crop_then_recover':
                img = random_crop_then_recover(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_translate':
                img = random_translate(img, return_distortion_type=self.return_distortion_type)
            elif distoration == 'random_rotate':
                img = random_rotate(img, return_distortion_type=self.return_distortion_type)
            return img
        
# jpeg_compression
# def jpeg_compression(img, quality_range=(10, 30), return_distortion_type=False):
def jpeg_compression(img, quality_range=(30, 90), return_distortion_type=False):
    img = np.array(img)
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(encimg, 1)
    img = Image.fromarray(img)
    if return_distortion_type:
        return img, f'quality_{quality}'
    else:
        return img

# add_gaussian_noise
# def add_gaussian_noise(img, std_range=(12.75, 102), return_distortion_type=False):
def add_gaussian_noise(img, std_range=(1, 5), return_distortion_type=False):
    img = np.array(img)
    mean = 0
    std = random.uniform(*std_range)
    g_noise = np.random.normal(mean, std, img.shape).astype('uint8')
    img = Image.fromarray(np.clip(img + g_noise, 0, 255))
    if return_distortion_type:
        return img, f'std_{std:.2f}'
    else:
        return img

# gaussian_blur
# def gaussian_blur(img, radius_choices=(2, 4, 6, 8, 10), return_distortion_type=False):
def gaussian_blur(img, radius_choices=(1, 2), return_distortion_type=False):
    radius = random.choice(radius_choices)
    img = img.filter(ImageFilter.GaussianBlur(radius))
    if return_distortion_type:
        return img, f'radius_{radius}'
    else:
        return img

# median_blur
# def median_blur(img, ksize_choices=(3, 7, 11, 15, 19), return_distortion_type=False):
def median_blur(img, ksize_choices=(3, 5, 7, 9), return_distortion_type=False):
    ksize = random.choice(ksize_choices)
    img = img.filter(ImageFilter.MedianFilter(ksize))
    if return_distortion_type:
        return img, f'ksize_{ksize}'
    else:
        return img
    
# salt_and_pepper_noise
# def salt_and_pepper_noise(img, ratio_range=(0.05, 0.4), return_distortion_type=False):
def salt_and_pepper_noise(img, ratio_range=(0.01, 0.07), return_distortion_type=False):
    img = np.array(img)
    s_vs_p = 0.5
    ratio = random.uniform(*ratio_range)
    out = np.copy(img)
    num_salt = np.ceil(ratio * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords[0], coords[1], :] = 1
    num_pepper = np.ceil(ratio * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    out[coords[0], coords[1], :] = 0
    img = Image.fromarray(out)
    if return_distortion_type:
        return img, f'ratio_{ratio:.2f}'
    else:
        return img
    
# resize_then_recover
# def resize_then_recover(img, ratio_range=(0.1, 0.9), return_distortion_type=False):
def resize_then_recover(img, ratio_range=(0.6, 0.9), return_distortion_type=False):
    img = np.array(img)
    height, width = img.shape[:2]
    h_ratio = random.uniform(*ratio_range)
    w_ratio = random.uniform(*ratio_range)
    new_height = int(height * h_ratio)
    new_width = int(width * w_ratio)
    img = cv2.resize(img, (new_width, new_height))
    img = cv2.resize(img, (width, height))
    img = Image.fromarray(img)
    if return_distortion_type:
        return img, f'ratio_h_{h_ratio:.2f}_w_{w_ratio:.2f}'
    else:
        return img

# brightness_transform
# def brightness_transform(img, factor_choices=(2, 4, 8, 12, 16), return_distortion_type=False):
def brightness_transform(img, factor_choices=(1, 2), return_distortion_type=False):
    factor = random.choice(factor_choices)
    img = transforms.ColorJitter(brightness=factor)(img)
    if return_distortion_type:
        return img, f'factor_{factor}'
    else:
        return img

###  for image content tamper 
def random_crop_with_ground_truth(img, crop_size_range=(0.1, 0.9)):
    width, height = img.size
    crop_width = random.randint(crop_size_range[0], crop_size_range[1])
    crop_height = random.randint(crop_size_range[0], crop_size_range[1])

    if crop_width > width or crop_height > height:
        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)
    
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    
    tampered_part = img.crop((left, top, right, bottom))
    img.paste(tampered_part, (left, top))
    
    tamper_loc = np.zeros((height, width), dtype=np.uint8)
    tamper_loc[top:bottom, left:right] = 1
    
    return img, tamper_loc

# Random Crop
# Randomly selects an area to crop and scales it back to its original dimensions
def random_crop_then_recover(img, crop_ratio_range=(0.7, 0.9), return_distortion_type=False):
    width, height = img.size
    
    # Randomly select crop ratio
    crop_ratio = random.uniform(*crop_ratio_range)
    target_width = int(width * crop_ratio)
    target_height = int(height * crop_ratio)
    
    # Randomly select the starting point for cropping
    left = random.randint(0, width - target_width)
    top = random.randint(0, height - target_height)
    right = left + target_width
    bottom = top + target_height
    
    # Crop Image
    cropped_img = img.crop((left, top, right, bottom))
    
    # Resize to original dimensions
    img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)
    
    if return_distortion_type:
        return img, f'ratio_{crop_ratio:.2f}'
    else:
        return img

# Random Shift
# Shifts the image horizontally and/or vertically, filling any parts shifted beyond the boundaries with black.
def random_translate(img, max_shift_ratio=(0.3, 0.3), return_distortion_type=False):
    width, height = img.size
    
    # Randomly select translation amount
    max_shift_x = int(width * max_shift_ratio[0])
    max_shift_y = int(height * max_shift_ratio[1])
    
    shift_x = random.randint(-max_shift_x, max_shift_x)
    shift_y = random.randint(-max_shift_y, max_shift_y)
    
    # Translation Using Affine Transformations
    # M is the translation matrix 
    # [1 0 tx]
    # [0 1 ty]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    img_cv = np.array(img)
    # cv2.warpAffine: Image transformation with boundary filling set to black (0) by default
    img_shifted = cv2.warpAffine(img_cv, M, (width, height))
    
    img = Image.fromarray(img_shifted)
    
    if return_distortion_type:
        return img, f'shift_x_{shift_x}_y_{shift_y}'
    else:
        return img

# Random Rotate
# Randomly rotate images while preserving pixel sharpness using nearest-neighbor interpolation
def random_rotate(img, angle_range=(-30, 30), return_distortion_type=False):
    angle = random.uniform(*angle_range)
    # Use PIL's rotate method with expand=False to preserve original dimensions and fill edges with black.
    img = img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(0, 0, 0))
    if return_distortion_type:
        return img, f'angle_{angle:.2f}'
    else:
        return img
    
if __name__ == '__main__':
    pass