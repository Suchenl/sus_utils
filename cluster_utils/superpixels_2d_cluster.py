import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.color import label2rgb
from sklearn.cluster import KMeans
import torch

def superpixels_2d_cluster(data: np.ndarray, n_segments: int = 1000, compactness: float = 10.0) -> np.ndarray:
    """
    Segments a grayscale image using superpixels (SLIC) and K-Means clustering.

    Args:
        data (np.ndarray): The input 2D grayscale image (H, W).
        n_segments (int): The desired number of superpixels.
        compactness (float): Balances color proximity and space proximity. 
                               Higher values create more square-like superpixels.

    Returns:
        np.ndarray: A 2D binary segmentation mask, where 0 is background and 1 is foreground.
    """
    # convert torch → numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    assert data.ndim == 2, "Input must be a 2D matrix [H, W]."
    
    # --- Step 0: Normalization ---
    # Convert the image to float format with a range of [0, 1].
    # This is the recommended input format for the SLIC algorithm.
    data_float = img_as_float(data)
    
    # --- Step 1: Generate Superpixels ---
    # The slic function returns a label map, where each pixel's value
    # is the ID of the superpixel it belongs to.
    # start_label=1 ensures labels start from 1, avoiding confusion with a background label of 0.
    segments = slic(data_float, n_segments=n_segments, compactness=compactness, start_label=1, channel_axis=None)
    
    # --- Step 2: Extract Features for Each Superpixel ---
    # We use the mean grayscale intensity of each superpixel as its representative feature.
    unique_labels = np.unique(segments)
    superpixel_means = []
    
    for label in unique_labels:
        # Create a mask to find all pixels belonging to the current superpixel
        mask = (segments == label)
        # Calculate the mean intensity of these pixels and store it
        mean_intensity = data_float[mask].mean()
        superpixel_means.append(mean_intensity)
    
    # Convert the feature list into a NumPy array suitable for K-Means (shape: [n_superpixels, 1])
    features = np.array(superpixel_means).reshape(-1, 1)
    
    # --- Step 3: Cluster Superpixel Features with K-Means ---
    # We want to segment into two classes (e.g., foreground and background), so n_clusters=2.
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(features)
    
    # Get the cluster label (0 or 1) for each superpixel
    superpixel_labels = kmeans.labels_
    
    # --- Step 4: Create the Final Segmentation Mask ---
    # Create a blank mask with the same dimensions as the original image
    final_mask = np.zeros_like(data_float, dtype=int)
    
    # Iterate through each superpixel and fill the mask according to its cluster label
    for i, label in enumerate(unique_labels):
        final_mask[segments == label] = superpixel_labels[i]
        
    # --- Correct Cluster Labels (Ensure bright regions are 1, dark regions are 0) ---
    # The labels (0 and 1) assigned by K-Means are arbitrary. We check which cluster
    # center (mean intensity) is higher and ensure that this cluster is labeled as 1 (foreground).
    cluster_centers = kmeans.cluster_centers_
    if cluster_centers[0] > cluster_centers[1]:
        # If cluster 0 represents the brighter region, we flip the labels (0->1, 1->0).
        final_mask = 1 - final_mask
        
    return final_mask

# --- Example Usage ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Create a sample input image x (H, W)
    # Here, we create a 512x512 black background with objects on it.
    H, W = 512, 512
    x = np.zeros((H, W), dtype=np.uint8)
    
    # Add a brighter object
    x[100:250, 100:250] = 220
    # Create a "hole" inside the object to simulate your problem
    x[150:200, 150:200] = 50 
    # Add another, dimmer object
    x[300:450, 300:450] = 180
    
    # Add some random noise
    noise = np.random.randint(0, 30, size=(H,W), dtype=np.uint8)
    x = np.clip(x + noise, 0, 255).astype(np.uint8)

    # --- Call the segmentation function ---
    # You can adjust n_segments and compactness to see how they affect the result.
    segmented_mask = segment_with_superpixels(x, n_segments=400, compactness=20)

    # --- Visualize the Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes.ravel()

    ax[0].imshow(x, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display superpixel boundaries for visualization
    # We generate them again here just for the visualization plot
    slic_segments = slic(x, n_segments=400, compactness=20)
    ax[1].imshow(label2rgb(slic_segments, x, kind='avg'), cmap='gray')
    ax[1].set_title('Superpixel Segmentation')
    ax[1].axis('off')

    ax[2].imshow(segmented_mask, cmap='gray')
    ax[2].set_title('Final Segmentation Mask')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()