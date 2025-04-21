import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import logging
import os  # Needed for visualize_segmentation

# Configure logging
logger = logging.getLogger(__name__)

# --- Load YOLO Model ---
MODEL_PATH = "yolov8n.pt"  # Or provide an absolute path / path relative to project root
yolo_model = None
try:
    # Consider checking os.path.exists(MODEL_PATH) if path is not guaranteed
    yolo_model = YOLO(MODEL_PATH)
    logger.info(f"YOLOv8 model loaded successfully from {MODEL_PATH}.")
except Exception as e:
    logger.error(f"Failed to load YOLO model from {MODEL_PATH}: {e}", exc_info=True)
    yolo_model = None


def perform_segmentation(image: np.ndarray):
    """
    Performs segmentation on the input image using thresholding and contour filtering.
    Optionally enhances with YOLO detection (if model loaded).

    Args:
        image: A NumPy array representing the preprocessed image (BGR format).

    Returns:
        A dictionary containing segmentation results: {'mask': binary_mask, 'contours': list_of_contours}
        Returns None if processing fails.
    """
    if image is None or image.size == 0:
        logger.error("perform_segmentation received an invalid or empty image.")
        return None
    logger.info(f"Starting segmentation for image with shape: {image.shape}, dtype: {image.dtype}")
    try:
        # --- Basic Segmentation (Thresholding & Contours) ---
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image  # Already grayscale
        else:
            logger.error(f"Unsupported image shape for grayscale conversion: {image.shape}")
            return None

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # --- Parameters to Tune ---
        block_size = 25  # Must be odd, > 1
        C = 4
        mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, C)

        kernel_size = 5
        iterations_close = 1
        iterations_open = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations_close)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=iterations_open)

        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 300
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        logger.info(f"Filtered contours by area (> {min_contour_area}): {len(filtered_contours)} remaining.")

        # Create a final mask from filtered contours
        filtered_mask = np.zeros_like(mask_opened)
        cv2.drawContours(filtered_mask, filtered_contours, -1, 255, -1)  # Draw filled contours

        segmentation_output = {
            'mask': filtered_mask,
            'contours': filtered_contours,
        }
        logger.info("Segmentation completed successfully.")
        return segmentation_output

    except cv2.error as cv_err:
        logger.error(f"OpenCV error during segmentation: {cv_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during segmentation: {e}", exc_info=True)
        return None


def visualize_segmentation(original_image: np.ndarray, segmentation_data: dict, output_path: str):
    """
    Draws the segmentation mask overlay onto the original image and saves it.

    Args:
        original_image: The original BGR image (NumPy array).
        segmentation_data: The dictionary returned by perform_segmentation (must contain 'mask').
        output_path: The full path where the visualization image will be saved.

    Returns:
        bool: True if visualization was saved successfully, False otherwise.
    """
    if original_image is None or segmentation_data is None or 'mask' not in segmentation_data:
        logger.error("visualize_segmentation: Invalid input image or segmentation data.")
        return False
    if not output_path:
        logger.error("visualize_segmentation: Output path not provided.")
        return False

    mask = segmentation_data['mask']
    if mask is None or mask.shape[:2] != original_image.shape[:2]:
        logger.error("visualize_segmentation: Mask is invalid or dimensions mismatch.")
        return False

    try:
        # Create a color overlay for the mask (e.g., semi-transparent blue)
        color_mask = np.zeros_like(original_image)
        color_mask[mask == 255] = [255, 0, 0]  # Blue color for mask area

        # Blend the original image and the color mask
        alpha = 0.4  # Transparency factor
        overlay_image = cv2.addWeighted(original_image, 1, color_mask, alpha, 0)

        # Optionally draw contours
        contours = segmentation_data.get('contours')
        if contours:
            cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 1)  # Green contours

        # Save the result
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                logger.error(f"Could not create directory {output_dir}: {e}")
                return False

        success = cv2.imwrite(output_path, overlay_image)
        if success:
            logger.info(f"Segmentation visualization saved to: {output_path}")
            return True
        else:
            logger.error(f"Failed to save segmentation visualization to: {output_path}")
            return False

    except cv2.error as cv_err:
        logger.error(f"OpenCV error during segmentation visualization: {cv_err}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during segmentation visualization: {e}", exc_info=True)
        return False