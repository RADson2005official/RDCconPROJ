import os
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self, image_dir):
        if not os.path.isdir(image_dir):
            logger.warning(f"Provided image directory does not exist: {image_dir}")
            # Decide if this should be a fatal error or just a warning
            # raise ValueError(f"Image directory not found: {image_dir}")
        self.image_dir = image_dir
        logger.info(f"ImageLoader initialized with directory: {self.image_dir}")
        self.top_images = []
        self.front_images = []
        self.current_index = 0
        self.load_images()

    def load_images(self):
        """Load images from the image directory"""
        logger.debug(f"Loading images from: {self.image_dir}")

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Image directory not found: {self.image_dir}")

        # Get all image files from the directory
        try:
            image_files = [f for f in os.listdir(self.image_dir) if self._is_image_file(f)]
            image_files.sort()  # Sort files alphabetically

            if not image_files:
                raise RuntimeError(f"No image files found in {self.image_dir}")

            logger.info(f"Found {len(image_files)} images: {image_files}")

            # Process files in pairs
            for i in range(0, len(image_files) - 1, 2):
                # Load image pair
                img1_path = os.path.join(self.image_dir, image_files[i])
                img2_path = os.path.join(self.image_dir, image_files[i + 1])

                img1 = self.load_image(img1_path)
                img2 = self.load_image(img2_path)

                if img1 is not None and img2 is not None:
                    self.top_images.append(img1)
                    self.front_images.append(img2)
                    logger.info(f"Loaded pair: {image_files[i]} and {image_files[i + 1]}")
                else:
                    logger.warning(f"Failed to load images: {image_files[i]} or {image_files[i + 1]}")

            if not self.top_images:
                raise RuntimeError("No valid image pairs were loaded")

            logger.info(f"Successfully loaded {len(self.top_images)} image pairs")

        except Exception as e:
            logger.error(f"Error loading images: {str(e)}")
            raise RuntimeError(f"Failed to load images: {str(e)}")

    def load_image(self, image_filepath: str):
        """
        Loads an image from the specified full filepath.

        Args:
            image_filepath: The absolute path to the image file.

        Returns:
            A NumPy array representing the loaded image (BGR format),
            or None if loading fails.
        """
        if not image_filepath:
            logger.error("load_image received an empty filepath.")
            return None

        if not os.path.exists(image_filepath):
            logger.error(f"Image file not found: {image_filepath}")
            return None
        if not os.path.isfile(image_filepath):
            logger.error(f"Path exists but is not a file: {image_filepath}")
            return None

        try:
            image = cv2.imread(image_filepath)

            if image is None:
                logger.error(f"Failed to load image (cv2.imread returned None): {image_filepath}")
                return None

            logger.info(f"Image loaded successfully: {image_filepath}, shape: {image.shape}")
            return image
        except cv2.error as cv_err:
            logger.error(f"OpenCV error loading image {image_filepath}: {cv_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading image {image_filepath}: {e}", exc_info=True)
            return None

    def _is_image_file(self, filename):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    def get_next_pair(self):
        """Get next pair of images, return None if no more images"""
        if self.current_index >= len(self.top_images):
            return None, None

        top_img = self.top_images[self.current_index]
        front_img = self.front_images[self.current_index]
        self.current_index += 1

        return top_img, front_img

    def reset(self):
        """Reset to first image pair"""
        self.current_index = 0
