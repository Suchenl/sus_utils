#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2025 Yuzhuo Chen (Suchenl)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This file contains a series of functions for generating 2D image transformation matrices, 
including affine transformations (translation, scaling, rotation, shearing, reflection) and perspective transformations. 

All transformations are relative to the image center (if applicable).

Additionally, a 'RandomProjectiveTransformationMatrixGenerator' class is implemented to randomly 
combine these transformations to generate a uniform, random projective transformation matrix.

The main block (if __name__ == "__main__") serves as a verification script. 
It works as follows:
1. An initial test image with four marked corners (TL, TR, BR, BL) is created.
2. For each individual transformation type, it:
   - Sets fixed, explicit parameters.
   - Generates the transformation matrix.
   - Prints both the parameters and the resulting matrix to the console.
   - Applies the transformation to the test image and saves the result as a PNG file.
3. It then tests the random generator class by creating and applying a combined random transformation.
4. All output images are saved to the 'outputs/check_transform_matrix_generator' directory, 
   allowing for easy visual and numerical verification of each function's correctness.
"""

import torch
import math
import cv2
import numpy as np
import random
from typing import Dict, Callable, Literal, Tuple, Union, Optional


def get_identity_matrix(**_) -> torch.Tensor:
    """
    Generates a 3x3 identity matrix.
    This matrix represents a transformation that does nothing.

    Returns:
        torch.Tensor: A 3x3 identity matrix.
    """
    return torch.eye(3, dtype=torch.float32)


def get_translate_matrix(H: int, W: int, tx: float = 0.0, ty: float = 0.0, **_) -> torch.Tensor:
    """
    Generates a 3x3 translation matrix.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        tx (float): The translation factor along the x-axis, expressed as a
                    fraction of the image width. A value of 0.5 means a translation
                    of W/2 pixels to the right. A negative value translates to the left.
        ty (float): The translation factor along the y-axis, expressed as a
                    fraction of the image height. A value of 0.5 means a translation
                    of H/2 pixels downwards. A negative value translates upwards.

    Returns:
        torch.Tensor: A 3x3 translation matrix.
    """
    return torch.tensor([
        [1, 0, tx * W],
        [0, 1, ty * H],
        [0, 0, 1]
    ], dtype=torch.float32)


def get_scale_matrix(H: int, W: int, scale_x: float = 1.0, scale_y: float = 1.0, **_) -> torch.Tensor:
    """
    Generates a 3x3 scaling matrix that scales relative to the image center.

    To scale around the center, the transformation is a composition of:
    1. Translate the image center to the origin.
    2. Apply the scaling.
    3. Translate the origin back to the image center.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        scale_x (float): The scaling factor along the x-axis. >1 zooms in, <1 zooms out.
        scale_y (float): The scaling factor along the y-axis. >1 zooms in, <1 zooms out.

    Returns:
        torch.Tensor: A 3x3 center-scaling matrix.
    """
    # Image center coordinates
    center_x = (W - 1) / 2
    center_y = (H - 1) / 2
    
    # The translation part of the matrix that results from composing T_inv @ S @ T
    tx = center_x * (1 - scale_x)
    ty = center_y * (1 - scale_y)

    return torch.tensor([
        [scale_x, 0,       tx],
        [0,       scale_y, ty],
        [0,       0,       1]
    ], dtype=torch.float32)


def get_rotate_matrix(H: int, W: int, angle_deg: float = 0.0, **_) -> torch.Tensor:
    """
    Generates a 3x3 rotation matrix that rotates relative to the image center.

    This is achieved by composing:
    1. Translate the image center to the origin.
    2. Apply the rotation.
    3. Translate the origin back to the image center.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        angle_deg (float): The rotation angle in degrees. Positive values result
                           in a counter-clockwise rotation.

    Returns:
        torch.Tensor: A 3x3 center-rotation matrix.
    """
    # Image center coordinates
    center_x = (W - 1) / 2
    center_y = (H - 1) / 2
    
    # Convert angle to radians
    theta = math.radians(angle_deg)
    
    # Pre-calculate cosine and sine
    c = math.cos(theta)
    s = math.sin(theta)
    
    # The translation part of the matrix that results from composing T_inv @ R @ T
    tx = center_x * (1 - c) + center_y * s
    ty = center_y * (1 - c) - center_x * s
    
    return torch.tensor([
        [c, -s, tx],
        [s,  c, ty],
        [0,  0, 1]
    ], dtype=torch.float32)


def get_shear_matrix(H: int, W: int, shear_x: float = 0.0, shear_y: float = 0.0, **_) -> torch.Tensor:
    """
    Generates a 3x3 shear matrix that shears relative to the image center.

    This ensures that the center of the image remains stationary during the shear.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        shear_x (float): The shear factor along the x-axis (horizontal shear).
        shear_y (float): The shear factor along the y-axis (vertical shear).

    Returns:
        torch.Tensor: A 3x3 center-shearing matrix.
    """
    # Image center coordinates
    center_x = (W - 1) / 2
    center_y = (H - 1) / 2
    
    # Translation components to keep the center fixed
    tx = -shear_x * center_y
    ty = -shear_y * center_x

    return torch.tensor([
        [1,       shear_x, tx],
        [shear_y, 1,       ty],
        [0,       0,       1]
    ], dtype=torch.float32)


def get_reflection_matrix(H: int, W: int, direction: Literal['horizontal', 'vertical', 'both'] = 'horizontal', **_) -> torch.Tensor:
    """
    Generates a 3x3 reflection (flip) matrix relative to the image center.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        direction (str): The axis of reflection. Can be 'horizontal' (left-right flip) 
                         ,'vertical' (top-bottom flip) or 'both' (both flips).

    Returns:
        torch.Tensor: A 3x3 center-reflection matrix.
    """
    if direction == 'horizontal':
        # Reflects across the vertical centerline of the image
        return torch.tensor([
            [-1, 0, W - 1],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=torch.float32)
    elif direction == 'vertical':
        # Reflects across the horizontal centerline of the image
        return torch.tensor([
            [1,  0, 0],
            [0, -1, H - 1],
            [0,  0, 1]
        ], dtype=torch.float32)
    elif direction == 'both':
        # Reflects across both centerlines (i.e., a 180-degree rotation)
        return torch.tensor([
            [-1, 0, W - 1],
            [ 0, -1, H - 1],
            [ 0,  0, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Invalid direction='{direction}'. Must be 'horizontal' or 'vertical'.")


def get_perspective_view_matrix(
    H: int, 
    W: int, 
    strength: float = 0.3,
    eye_pos: Literal['top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right'] = 'top',
    mode: Literal['outset', 'inset'] = "outset",
    **_
) -> torch.Tensor:
    """
    A wrapper for the internal perspective function to generate a single forward matrix.
    
    This function computes a perspective transformation by mapping the four corners of
    the image to new "distorted" positions, simulating a change in viewpoint.

    Args:
        H (int): The height of the image.
        W (int): The width of the image.
        strength (float): A value > 0 controlling the intensity of the perspective
                          distortion. It's a fraction of the image width/height.
        eye_pos (str): The simulated viewpoint position, which determines which
                       corners of the image are moved.
        mode (str): Determines the direction of corner movement.
                    - "outset": Moves corners outwards, widening the edge.
                    - "inset": Moves corners inwards, narrowing the edge.

    Returns:
        torch.Tensor: The forward 3x3 perspective transformation matrix.
    """
    # This internal function returns a tuple (M, M_inv). We only need the forward matrix M.
    M, _ = _get_perspective_view_matrix_internal(H, W, strength, eye_pos, mode)
    return M


def _get_perspective_view_matrix_internal(
    H: int, W: int, strength: float, eye_pos: str, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal implementation using OpenCV to calculate perspective matrices."""
    # Source coordinates: the four corners of the original image.
    src_points = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    # Destination coordinates start as a copy and are then modified.
    dst_points = src_points.copy()

    # Calculate the margin of movement based on strength.
    margin_x = W * strength
    margin_y = H * strength
    
    # Determine the direction of movement.
    sign = -1.0 if mode == "outset" else 1.0

    # Modify the destination corner points based on the viewpoint.
    if eye_pos == "top":
        dst_points[0, 0] += sign * margin_x   # Top-left corner, x-coordinate
        dst_points[1, 0] -= sign * margin_x   # Top-right corner, x-coordinate
    elif eye_pos == "bottom":
        dst_points[2, 0] -= sign * margin_x   # Bottom-right corner, x-coordinate
        dst_points[3, 0] += sign * margin_x   # Bottom-left corner, x-coordinate
    elif eye_pos == "left":
        dst_points[0, 1] += sign * margin_y   # Top-left corner, y-coordinate
        dst_points[3, 1] -= sign * margin_y   # Bottom-left corner, y-coordinate
    elif eye_pos == "right":
        dst_points[1, 1] += sign * margin_y   # Top-right corner, y-coordinate
        dst_points[2, 1] -= sign * margin_y   # Bottom-right corner, y-coordinate
    elif eye_pos == "top_left":
        dst_points[0] += np.array([sign * margin_x, sign * margin_y])
    elif eye_pos == "top_right":
        dst_points[1] += np.array([-sign * margin_x, sign * margin_y])
    elif eye_pos == "bottom_left":
        dst_points[3] += np.array([sign * margin_x, -sign * margin_y])
    elif eye_pos == "bottom_right":
        dst_points[2] += np.array([-sign * margin_x, -sign * margin_y])
    else:
        raise ValueError(f"Invalid eye_pos='{eye_pos}'")

    # Use OpenCV to compute the 3x3 perspective transformation matrix.
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Also compute the inverse transformation.
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    # Convert numpy arrays to PyTorch tensors.
    M_tensor = torch.from_numpy(M).to(torch.float32)
    M_inv_tensor = torch.from_numpy(M_inv).to(torch.float32)
    
    return M_tensor, M_inv_tensor

class RandomProjectiveTransformationMatrixGenerator:
    """
    Applies a random sequence of projective transformations.

    When called, this class randomly selects a specified number of transformation
    types (e.g., rotation, scaling, perspective), generates random parameters for
    each, computes their corresponding transformation matrices, and multiplies them
    to produce a single, composite transformation matrix.
    """
    def __init__(self,
                 n_transforms: Tuple[int, int] = (1, 4),
                 translation_range: Tuple[float, float] = (-0.1, 0.1),
                 scaling_range: Tuple[float, float] = (0.8, 1.2),
                 rotation_range: Tuple[float, float] = (-20, 20),
                 shear_range: Tuple[float, float] = (-0.1, 0.1),
                 use_reflection: bool = True,
                 perspective_strength_range: Tuple[float, float] = (0.05, 0.2),
                 print_transform_info=False):
        """
        Initializes the random transformation generator.

        Args:
            n_transforms (Tuple[int, int]): A tuple (min, max) specifying the range
                for the number of transformations to apply in each call.
            translation_range (Tuple[float, float]): Range for random translation factors.
            scaling_range (Tuple[float, float]): Range for random scaling factors.
            rotation_range (Tuple[float, float]): Range for random rotation angles in degrees.
            shear_range (Tuple[float, float]): Range for random shear factors.
            perspective_strength_range (Tuple[float, float]): Range for the strength
                of the perspective distortion.
        """
        assert 0 <= n_transforms[0] <= n_transforms[1], "n_transforms must be a valid range (min, max)"
        self.n_min, self.n_max = n_transforms

        # Store available transformation functions
        self.transform_functions = [
            get_translate_matrix,
            get_scale_matrix,
            get_rotate_matrix,
            get_shear_matrix,
            get_perspective_view_matrix
        ]
        if use_reflection:
            self.transform_functions.append(get_reflection_matrix)
        
        # Store parameter ranges
        self.param_ranges = {
            'translation': translation_range,
            'scaling': scaling_range,
            'rotation': rotation_range,
            'shear': shear_range,
            'perspective': perspective_strength_range
        }
        
        # Options for categorical parameters in perspective and reflection transforms
        self.perspective_eye_pos = ['top', 'bottom', 'left', 'right', 'top_left', 
                                    'top_right', 'bottom_left', 'bottom_right']
        self.perspective_mode = ['outset', 'inset']
        self.reflection_directions = ['horizontal', 'vertical', 'both']

        # Other configs
        self.print_transform_info = print_transform_info
        
    def _get_random_params(self, transform_func: Callable, generator: Optional[torch.Generator] = None) -> Dict:
        """Generates a dictionary of random parameters for a given transformation function."""
        def _uniform(range_tuple):
            min_val, max_val = range_tuple
            # Generate a random float in [0, 1) and scale it to the desired range
            return min_val + (max_val - min_val) * torch.rand(1, generator=generator, device=generator.device).item()
        
        if transform_func == get_translate_matrix:
            return {'tx': _uniform(self.param_ranges['translation']), 
                    'ty': _uniform(self.param_ranges['translation'])}
        
        elif transform_func == get_scale_matrix:
            return {'scale_x': _uniform(self.param_ranges['scaling']), 
                    'scale_y': _uniform(self.param_ranges['scaling'])}
            
        elif transform_func == get_rotate_matrix:
            return {'angle_deg': _uniform(self.param_ranges['rotation'])}
            
        elif transform_func == get_shear_matrix:
            return {'shear_x': _uniform(self.param_ranges['shear']), 
                    'shear_y': _uniform(self.param_ranges['shear'])}

        elif transform_func == get_reflection_matrix:
            # ## DDP-SAFE CHANGE ##: Replace random.choice with torch.randint
            idx = torch.randint(0, len(self.reflection_directions), (1,), generator=generator, device=generator.device).item()
            return {'direction': self.reflection_directions[idx]}
            
        elif transform_func == get_perspective_view_matrix:
            # ## DDP-SAFE CHANGE ##: Replace multiple random.choice calls
            strength = _uniform(self.param_ranges['perspective'])
            eye_pos_idx = torch.randint(0, len(self.perspective_eye_pos), (1,), generator=generator, device=generator.device).item()
            mode_idx = torch.randint(0, len(self.perspective_mode), (1,), generator=generator, device=generator.device).item()
            return {'strength': strength, 
                    'eye_pos': self.perspective_eye_pos[eye_pos_idx], 
                    'mode': self.perspective_mode[mode_idx]}
        else:
            return {}

    def __call__(self, H: int, W: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Generates and combines random transformation matrices.

        Args:
            H (int): The height of the image.
            W (int): The width of the image.

        Returns:
            torch.Tensor: A single 3x3 matrix representing the combined
                          random projective transformations.
        """
        # Start with an identity matrix
        final_matrix = get_identity_matrix()

        # Determine how many transformations to apply
        if self.n_max == 0:
            return final_matrix
        
        n_to_apply = torch.randint(self.n_min, self.n_max + 1, (1,), generator=generator, device=generator.device).item()
        
        if n_to_apply == 0:
            return final_matrix

        # ## DDP-SAFE CHANGE ##: Replace random.choices and random.sample
        num_available_funcs = len(self.transform_functions)
        if n_to_apply > num_available_funcs:
            # Sample with replacement
            # Use torch.randint for sampling with replacement (like random.choices)
            chosen_indices = torch.randint(0, num_available_funcs, (n_to_apply,), generator=generator, device=generator.device)
            chosen_funcs = [self.transform_functions[i] for i in chosen_indices]
        else:
            # Sample without replacement
            # Use torch.randperm for sampling without replacement (like random.sample)
            indices = torch.randperm(num_available_funcs, generator=generator, device=generator.device)
            chosen_funcs = [self.transform_functions[i] for i in indices[:n_to_apply]]

        # Generate and apply each transformation
        for func in chosen_funcs:
            # Get random parameters for the chosen function
            params = self._get_random_params(func, generator=generator)
            
            # Generate the transformation matrix
            transform_matrix = func(H=H, W=W, **params)
            
            if self.print_transform_info:
                print("Transformation:", func.__name__, "\tParams:", params)
            
            # Combine it with the previous transformations.
            # New transforms are applied first, so they are on the left (M_new @ M_old)
            final_matrix = torch.matmul(transform_matrix, final_matrix)
            
        return final_matrix


if __name__ == '__main__':    
    import os
    def create_test_image(H, W):
        """Creates a sample image with a rectangle and markers for visualization."""
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Inner rectangle coordinates
        x1, y1 = int(W * 0.2), int(H * 0.2)
        x2, y2 = int(W * 0.8), int(H * 0.8)
        # Draw a white rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
        # --- Mark all four corners ---
        # Top-Left (TL): Red
        cv2.circle(img, (x1, y1), 10, (0, 0, 255), -1)
        cv2.putText(img, 'TL', (x1 + 15, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Top-Right (TR): Green
        cv2.circle(img, (x2, y1), 10, (0, 255, 0), -1)
        cv2.putText(img, 'TR', (x2 - 45, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Bottom-Right (BR): Blue
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)
        cv2.putText(img, 'BR', (x2 - 45, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Bottom-Left (BL): Yellow
        cv2.circle(img, (x1, y2), 10, (0, 255, 255), -1)
        cv2.putText(img, 'BL', (x1 + 15, y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return img

    # --- Setup ---
    H, W = 400, 600
    output_dir = "outputs/check_transform_matrix_generator"
    os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist
    
    original_image = create_test_image(H, W)
    original_image_path = os.path.join(output_dir, "00_original.png")
    cv2.imwrite(original_image_path, original_image)
    print(f"Saved original image to: {original_image_path}")
    
    dsize = (W, H)
    only_test_random_transform_generator = True

    if not only_test_random_transform_generator:
        # --- 1. Test Translation ---
        print("\n--- 1. Test Translation ---")
        params = {'tx': 0.2, 'ty': 0.1}
        print(f"Parameters: {params}")
        M_translate = get_translate_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_translate)
        img_translated = cv2.warpAffine(original_image, M_translate[:2, :], dsize)
        cv2.imwrite(os.path.join(output_dir, "01_translated.png"), img_translated)

        # --- 2. Test Scaling ---
        print("\n--- 2. Test Scaling ---")
        params = {'scale_x': 0.7, 'scale_y': 1.2}
        print(f"Parameters: {params}")
        M_scale = get_scale_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_scale)
        img_scaled = cv2.warpAffine(original_image, M_scale[:2, :], dsize)
        cv2.imwrite(os.path.join(output_dir, "02_scaled.png"), img_scaled)
        
        # --- 3. Test Rotation ---
        print("\n--- 3. Test Rotation ---")
        params = {'angle_deg': 30}
        print(f"Parameters: {params}")
        M_rotate = get_rotate_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_rotate)
        img_rotated = cv2.warpAffine(original_image, M_rotate[:2, :], dsize)
        cv2.imwrite(os.path.join(output_dir, "03_rotated.png"), img_rotated)
        
        # --- 4. Test Shear ---
        print("\n--- 4. Test Shear ---")
        params = {'shear_x': 0.4, 'shear_y': -0.1}
        print(f"Parameters: {params}")
        M_shear = get_shear_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_shear)
        img_sheared = cv2.warpAffine(original_image, M_shear[:2, :], dsize)
        cv2.imwrite(os.path.join(output_dir, "04_sheared.png"), img_sheared)

        # --- 5. Test Reflection ---
        print("\n--- 5. Test Reflection ---")
        params = {'direction': 'both'}
        print(f"Parameters: {params}")
        M_reflect = get_reflection_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_reflect)
        img_reflected = cv2.warpAffine(original_image, M_reflect[:2, :], dsize)
        cv2.imwrite(os.path.join(output_dir, "05_reflected.png"), img_reflected)

        # --- 6. Test Perspective ---
        print("\n--- 6. Test Perspective ---")
        params = {'strength': 0.3, 'eye_pos': 'bottom', 'mode': 'inset'}
        print(f"Parameters: {params}")
        M_perspective = get_perspective_view_matrix(H, W, **params).numpy()
        print("Matrix:\n", M_perspective)
        img_perspective = cv2.warpPerspective(original_image, M_perspective, dsize)
        cv2.imwrite(os.path.join(output_dir, "06_perspective.png"), img_perspective)
    
    # --- 7. Test RandomProjectiveTransformationMatrixGenerator Class ---
    print("\n--- 7. Test Random Transformation Class ---")
    random_transformer = RandomProjectiveTransformationMatrixGenerator(
        n_transforms=(3, 6),
        translation_range=(-0.7, 0.7),
        scaling_range=(0.3, 1.5),
        rotation_range=(-90, 90),
        shear_range=(-0.3, 0.3),
        perspective_strength_range=(0.05, 0.5)
        )
    M_random = random_transformer(H, W).numpy()
    print("Generated Random Matrix:\n", M_random)
    img_random = cv2.warpPerspective(original_image, M_random, dsize)
    cv2.imwrite(os.path.join(output_dir, "07_random_combined.png"), img_random)

    print(f"\nVerification finished. All images are saved in the '{output_dir}' directory.")