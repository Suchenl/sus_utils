from PIL import Image

def crop_image(image: Image.Image,
               to_square: bool = False,
               center_crop: bool = True,
               start_coords: tuple[int, int] = (0, 0),
               ) -> Image.Image:
    if to_square:
        min_dim = min(image.size)
        if center_crop:
            left = (image.width - min_dim) // 2
            top = (image.height - min_dim) // 2
            right = (image.width + min_dim) // 2
            bottom = (image.height + min_dim) // 2
        else:
            left, top, right, bottom = start_coords[0], start_coords[1], start_coords[0] + min_dim, start_coords[1] + min_dim
    else:
        raise NotImplementedError("Currently only to_square cropping is implemented.")
    
    return image.crop((left, top, right, bottom))