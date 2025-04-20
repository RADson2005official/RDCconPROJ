import os
import cv2
import logging
import numpy as np  # Fixed syntax error here


class ImageLoader:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.top_images = []
        self.front_images = []
        self.current_index = 0
        self.load_images()

    def load_images(self):
        """Load images from the image directory"""
        logging.debug(f"Loading images from: {self.image_dir}")

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Image directory not found: {self.image_dir}")

        # Get all image files from the directory
        try:
            image_files = [f for f in os.listdir(self.image_dir) if self._is_image_file(f)]
            image_files.sort()  # Sort files alphabetically

            if not image_files:
                raise RuntimeError(f"No image files found in {self.image_dir}")

            logging.info(f"Found {len(image_files)} images: {image_files}")

            # Process files in pairs
            for i in range(0, len(image_files) - 1, 2):
                # Load image pair
                img1_path = os.path.join(self.image_dir, image_files[i])
                img2_path = os.path.join(self.image_dir, image_files[i + 1])

                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)

                if img1 is not None and img2 is not None:
                    self.top_images.append(img1)
                    self.front_images.append(img2)
                    logging.info(f"Loaded pair: {image_files[i]} and {image_files[i + 1]}")
                else:
                    logging.warning(f"Failed to load images: {image_files[i]} or {image_files[i + 1]}")

            if not self.top_images:
                raise RuntimeError("No valid image pairs were loaded")

            logging.info(f"Successfully loaded {len(self.top_images)} image pairs")

        except Exception as e:
            logging.error(f"Error loading images: {str(e)}")
            raise RuntimeError(f"Failed to load images: {str(e)}")

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
