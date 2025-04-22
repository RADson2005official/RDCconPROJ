import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('Agg')  # Use Agg backend since we're not using interactive GUI


def process_with_canny(image, low_threshold=50, high_threshold=150, aperture_size=3):
    """
    Process image using Canny edge detection

    Parameters:
    - image: Input image
    - low_threshold: Lower threshold for the hysteresis procedure
    - high_threshold: Upper threshold for the hysteresis procedure
    - aperture_size: Aperture size for the Sobel operator

    Returns:
    - edges: Edge-detected image
    - processing_time: Time taken for processing
    - aperture_size: Aperture size used (potentially adjusted)
    """
    start_time = time.time()

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise (particularly important for sand/gravel textures)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)

    # Check if edges are detected properly, if not, adjust aperture size
    edge_pixels = np.count_nonzero(edges)
    edge_percentage = edge_pixels / (edges.shape[0] * edges.shape[1])

    # If very few edges are detected (less than 1% of image), try with aperture_size=5
    if edge_percentage < 0.01 and aperture_size == 3:
        aperture_size = 5
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=aperture_size)

        # If still few edges, try with modified thresholds
        edge_pixels = np.count_nonzero(edges)
        edge_percentage = edge_pixels / (edges.shape[0] * edges.shape[1])
        if edge_percentage < 0.01:
            # Try with lower thresholds
            edges = cv2.Canny(blurred, low_threshold * 0.5, high_threshold * 0.8, apertureSize=aperture_size)

    processing_time = time.time() - start_time

    return edges, processing_time, aperture_size


def save_edge_detection_result(original_image, edge_image, output_path):
    """
    Save edge detection result as comparison image

    Parameters:
    - original_image: Original input image
    - edge_image: Edge-detected image
    - output_path: Path to save the result

    Returns:
    - bool: Success status
    """
    try:
        # Create figure for matplotlib
        fig = Figure(figsize=(10, 6))

        # Add subplots
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Display original image
        if len(original_image.shape) == 3:
            ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(original_image, cmap='gray')

        ax1.set_title("Original Image")
        ax1.axis('off')

        # Display edge image
        ax2.imshow(edge_image, cmap='gray')
        ax2.set_title("Edge Detection")
        ax2.axis('off')

        fig.tight_layout()
        fig.savefig(output_path)

        return True
    except Exception as e:
        print(f"Error saving edge detection result: {e}")
        return False


def process_image_edge_detection(image_path, output_folder='output', low_threshold=150, high_threshold=120,
                                 aperture_size=3):
    """
    Process an image with edge detection and save results

    Parameters:
    - image_path: Path to input image
    - output_folder: Folder to save results
    - low_threshold: Lower threshold for Canny
    - high_threshold: Upper threshold for Canny
    - aperture_size: Aperture size for Sobel operator

    Returns:
    - dict: Results including paths to saved images and processing details
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load image
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        return {'error': f"Could not load image: {image_path}"}

    # Get base filename
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    # Process with Canny using provided parameters
    edge_image, processing_time, aperture_used = process_with_canny(
        cv_image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        aperture_size=aperture_size
    )

    # Save edge image
    edge_path = os.path.join(output_folder, f"{name}_edges{ext}")
    cv2.imwrite(edge_path, edge_image)

    # Save original image (added from edge.py)
    original_path = os.path.join(output_folder, f"{name}_original{ext}")
    cv2.imwrite(original_path, cv_image)

    # Save comparison image
    comparison_path = os.path.join(output_folder, f"{name}_comparison.jpg")
    save_edge_detection_result(cv_image, edge_image, comparison_path)

    return {
        'original_image_path': original_path,
        'edge_image_path': edge_path,
        'comparison_path': comparison_path,
        'processing_time': processing_time,
        'aperture_size_used': aperture_used
    }