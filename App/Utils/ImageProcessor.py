import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
from PIL import Image
from torchvision import transforms


class ImageProcessor:
    def __init__(self):
        logging.getLogger(__name__).setLevel(logging.INFO)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_dl = False  # Default to traditional methods
        self.edge_model = None
        self.material_classifier = None
        self.feature_extractor = None
        self.material_classifier_initialized = False

    def initialize_dl_model(self):
        if self.edge_model is not None:
            return

        try:
            logging.info("Initializing deep learning model...")
            # Try offline model first
            model_path = Path(__file__).parent / 'models' / 'deeplabv3_resnet50.pth'
            if model_path.exists():
                # Add safe globals for model loading
                torch.serialization.add_safe_globals(['DeepLabV3'])
                self.edge_model = torch.load(
                    model_path,
                    weights_only=True,  # Load only weights to reduce memory
                    map_location=self.device
                )
            else:
                self.edge_model = torch.hub.load(
                    'pytorch/vision:v0.10.0',
                    'deeplabv3_resnet50',
                    pretrained=True,
                    force_reload=False,
                    trust_repo=True
                )
                # Save model for offline use
                os.makedirs(model_path.parent, exist_ok=True)
                torch.save(self.edge_model, model_path, _use_new_zipfile_serialization=False)

            self.edge_model.to(self.device)
            self.edge_model.eval()
            self.use_dl = True
            logging.info("Deep learning model initialized successfully")
        except Exception as e:
            self.use_dl = False
            logging.warning(f"Deep learning model initialization failed: {str(e)}")
            logging.info("Falling back to traditional methods")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def initialize_material_classifier(self):
        if self.material_classifier is not None:
            return

        try:
            logging.info("Initializing material classifier...")
            model_name = 'google/vit-base-patch16-224'
            model_path = Path(__file__).parent / 'models' / 'material_classifier'

            # Create models directory if it doesn't exist
            os.makedirs(model_path.parent, exist_ok=True)

            try:
                # Configure model parameters
                model_config = {
                    'num_labels': 3,
                    'id2label': {0: "Sand", 1: "Gravel", 2: "Other"},
                    'label2id': {"Sand": 0, "Gravel": 1, "Other": 2},
                    'ignore_mismatched_sizes': True,
                    'low_cpu_mem_usage': True  # Enable memory optimization
                }

                # Initialize the classifier head for our custom classes
                def init_classifier_head(model):
                    model.classifier = torch.nn.Linear(model.config.hidden_size, 3)
                    return model

                if model_path.exists():
                    logging.info("Loading model from local path...")
                    try:
                        self.feature_extractor = ViTImageProcessor.from_pretrained(
                            str(model_path),
                            local_files_only=True
                        )
                        self.material_classifier = ViTForImageClassification.from_pretrained(
                            str(model_path),
                            local_files_only=True,
                            **model_config
                        )
                    except Exception as local_error:
                        logging.warning(f"Failed to load local model: {str(local_error)}")
                        if os.path.exists(str(model_path)):
                            import shutil
                            shutil.rmtree(str(model_path))
                        raise

                if not model_path.exists():
                    logging.info("Downloading pre-trained model...")
                    self.feature_extractor = ViTImageProcessor.from_pretrained(
                        model_name
                    )
                    # Load base model and reinitialize classifier head
                    base_model = ViTForImageClassification.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=True  # Enable memory optimization
                    )
                    self.material_classifier = init_classifier_head(base_model)
                    # Update config after head reinitialization
                    self.material_classifier.config.update(model_config)

                    # Save models for offline use
                    logging.info("Saving model for offline use...")
                    os.makedirs(str(model_path), exist_ok=True)
                    self.feature_extractor.save_pretrained(str(model_path))
                    self.material_classifier.save_pretrained(str(model_path))

                self.material_classifier.to(self.device)
                self.material_classifier.eval()
                self.material_classifier_initialized = True
                logging.info("Material classifier initialized successfully")
            except Exception as model_error:
                logging.error(f"Model loading error: {str(model_error)}")
                raise
        except Exception as e:
            self.material_classifier_initialized = False
            logging.error(f"Material classifier initialization failed: {str(e)}")
            logging.warning("Material classification will be disabled")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def preprocess_image(self, image):
        # Input validation
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image")

        # Convert to grayscale and apply noise reduction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)

        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Apply Gaussian blur for edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        return blurred

    def classify_material(self, image):
        if not hasattr(self, 'material_classifier_initialized') or not self.material_classifier_initialized:
            # Initialize classifier if not already done
            self.initialize_material_classifier()
            if not self.material_classifier_initialized:
                logging.warning("Material classifier not initialized, skipping classification")
                return "Unknown", 0.0

        try:
            # Input validation
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Invalid input image")
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be a 3-channel BGR image")

            # Convert BGR to RGB and resize
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (224, 224))
            pil_image = Image.fromarray(rgb_image)

            # Prepare image for ViT with normalization
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt", do_normalize=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get prediction
            with torch.no_grad():
                outputs = self.material_classifier(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()

                # Clear GPU memory immediately after prediction
                del outputs, logits, probabilities
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            material_types = {0: "Sand", 1: "Gravel", 2: "Other"}
            return material_types[predicted_class], confidence

        except ValueError as e:
            logging.error(f"Invalid input: {str(e)}")
            return "Unknown", 0.0
        except Exception as e:
            logging.error(f"Material classification failed: {str(e)}")
            return "Unknown", 0.0
        finally:
            # Clean up any GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def detect_edges(image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Use Canny edge detection with automatic threshold calculation
        median = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(denoised, lower, upper)

        # Enhance edges using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges, thresh

    @staticmethod
    def detect_corners(image, max_corners=4):
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=max_corners,
            qualityLevel=0.01,
            minDistance=100
        )
        if corners is not None:
            corners = np.float32(corners)
            corners = corners.reshape(-1, 2)
        return corners

    @staticmethod
    def sort_corners(corners):
        center = np.mean(corners, axis=0)
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        return corners[np.argsort(angles)]

    @staticmethod
    def enhance_edges(image):
        """Enhance edges for better corner detection"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return enhanced

    @staticmethod
    def get_cube_mask(image):
        """Extract cube mask using adaptive thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def detect_pile_edges(self, image):
        try:
            # Input validation
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Invalid input image")

            # Convert and enhance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            enhanced = self.enhance_edges(denoised)

            # Create visualization dictionary
            viz_steps = {
                'original': image.copy(),
                'grayscale': cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                'denoised': cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
                'enhanced': cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            }

            if self.use_dl:
                try:
                    # Initialize model if not already done
                    self.initialize_dl_model()

                    # Prepare for deep learning
                    input_tensor = torch.from_numpy(enhanced).float()
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                    input_tensor = F.interpolate(input_tensor, size=(224, 224))
                    input_tensor = input_tensor.to(self.device)

                    with torch.no_grad():
                        output = self.edge_model(input_tensor)['out'][0]
                        edges_dl = F.interpolate(output.unsqueeze(0),
                                                 size=image.shape[:2],
                                                 mode='bilinear').squeeze(0)
                        edges_dl = edges_dl.argmax(0).cpu().numpy()

                    # Traditional edge detection with adaptive thresholding
                    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
                    edges_cv = cv2.Canny(blur, 50, 150)

                    # Combine edges with morphological operations
                    edges = cv2.bitwise_and(edges_dl.astype(np.uint8), edges_cv)
                    kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                    # Add visualization steps
                    viz_steps['dl_edges'] = cv2.cvtColor(edges_dl.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
                    viz_steps['cv_edges'] = cv2.cvtColor(edges_cv, cv2.COLOR_GRAY2BGR)

                finally:
                    # Clean GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                # Enhanced traditional edge detection
                blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

                # Adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # Multi-scale edge detection
                edges_low = cv2.Canny(blur, 30, 100)
                edges_high = cv2.Canny(blur, 70, 200)
                edges = cv2.addWeighted(edges_low, 0.5, edges_high, 0.5, 0)

                # Combine with threshold
                edges = cv2.bitwise_and(edges, thresh)

                # Clean up edges
                kernel = np.ones((3, 3), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

                # Add visualization steps
                viz_steps['threshold'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                viz_steps['edges_low'] = cv2.cvtColor(edges_low, cv2.COLOR_GRAY2BGR)
                viz_steps['edges_high'] = cv2.cvtColor(edges_high, cv2.COLOR_GRAY2BGR)

            # Post-process edges
            processed_edges = self.post_process_edges(edges)

            # Create colored edge visualization
            edge_viz = image.copy()
            edge_viz[processed_edges > 0] = [0, 255, 0]  # Green color for detected edges

            # Add final visualizations
            viz_steps['detected_edges'] = edge_viz
            viz_steps['final_edges'] = cv2.cvtColor(processed_edges, cv2.COLOR_GRAY2BGR)

            # Display visualization steps
            self.display_edge_detection_steps(viz_steps)

            return processed_edges

        except Exception as e:
            logging.error(f"Edge detection failed: {str(e)}")
            raise

    def display_edge_detection_steps(self, viz_steps):
        """Display intermediate steps of edge detection process.

        Args:
            viz_steps (dict): Dictionary containing visualization images for each step
        """
        try:
            # Create a figure with subplots
            num_steps = len(viz_steps)
            cols = min(3, num_steps)  # Maximum 3 columns
            rows = (num_steps + cols - 1) // cols

            plt.figure(figsize=(15, 5 * rows))

            # Display each step
            for idx, (step_name, step_img) in enumerate(viz_steps.items(), 1):
                plt.subplot(rows, cols, idx)
                plt.imshow(cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB))
                plt.title(step_name.replace('_', ' ').title())
                plt.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Failed to display edge detection steps: {str(e)}")
        """Display the edge detection visualization steps."""
        try:
            # Create a figure with subplots
            rows = (len(viz_steps) + 2) // 3  # 3 images per row
            fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
            fig.suptitle('Edge Detection Steps', fontsize=16)

            # Flatten axes for easier iteration
            axes = axes.flatten() if rows > 1 else [axes]

            # Display each step
            for idx, (step_name, step_img) in enumerate(viz_steps.items()):
                axes[idx].imshow(cv2.cvtColor(step_img, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(step_name.replace('_', ' ').title())
                axes[idx].axis('off')

            # Hide empty subplots
            for idx in range(len(viz_steps), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.warning(f"Failed to display edge detection steps: {str(e)}")

    def post_process_edges(self, edges):
        # Post-processing
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    def process_image_pair(self, top_image, front_image):
        """Process both images and return enhanced results with edges and corners"""
        # Get pile edges
        top_edges = self.detect_pile_edges(top_image)
        front_edges = self.detect_pile_edges(front_image)

        # Enhance edges
        top_enhanced = self.enhance_edges(top_image)
        front_enhanced = self.enhance_edges(front_image)

        # Get masks
        top_mask = self.get_cube_mask(top_image)
        front_mask = self.get_cube_mask(front_image)

        # Detect corners with masks
        top_corners = self.detect_corners(top_enhanced * (top_mask // 255))
        front_corners = self.detect_corners(front_enhanced * (front_mask // 255))

        # Detect contours
        top_contours, _ = cv2.findContours(top_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        front_contours, _ = cv2.findContours(front_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get points along contours
        top_points = np.vstack([cont.reshape(-1, 2) for cont in top_contours]) if len(top_contours) > 0 else np.array(
            [])
        front_points = np.vstack([cont.reshape(-1, 2) for cont in front_contours]) if len(
            front_contours) > 0 else np.array([])

        # Visualize results
        self.visualize_detection_results(
            top_image, front_image,
            top_edges, front_edges,
            top_corners, front_corners,
            top_contours, front_contours
        )

        return {
            'top_points': top_points,
            'front_points': front_points,
            'top_edges': top_edges,
            'front_edges': front_edges,
            'top_corners': top_corners,
            'front_corners': front_corners,
            'top_enhanced': top_enhanced,
            'front_enhanced': front_enhanced,
            'top_mask': top_mask,
            'front_mask': front_mask
        }

    def visualize_detection_results(self, top_image, front_image, top_edges, front_edges,
                                    top_corners, front_corners, top_contours, front_contours):
        """Visualize detection results on original images."""
        try:
            # Create copies for visualization
            top_viz = top_image.copy()
            front_viz = front_image.copy()

            # Draw edges
            top_viz[top_edges > 0] = [0, 255, 0]  # Green edges
            front_viz[front_edges > 0] = [0, 255, 0]

            # Draw contours
            cv2.drawContours(top_viz, top_contours, -1, (255, 0, 0), 2)  # Blue contours
            cv2.drawContours(front_viz, front_contours, -1, (255, 0, 0), 2)

            # Draw corners
            if top_corners is not None:
                for corner in top_corners:
                    cv2.circle(top_viz, tuple(corner.astype(int)), 5, (0, 0, 255), -1)  # Red corners
            if front_corners is not None:
                for corner in front_corners:
                    cv2.circle(front_viz, tuple(corner.astype(int)), 5, (0, 0, 255), -1)

            # Display results
            plt.figure(figsize=(12, 6))

            plt.subplot(121)
            plt.imshow(cv2.cvtColor(top_viz, cv2.COLOR_BGR2RGB))
            plt.title('Top View Detection Results')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(cv2.cvtColor(front_viz, cv2.COLOR_BGR2RGB))
            plt.title('Front View Detection Results')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.warning(f"Failed to visualize detection results: {str(e)}")

    def create_object_mask(self, image):
        """Create a binary mask for object segmentation using adaptive thresholding and contour detection."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply adaptive thresholding
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )

            # Find contours
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Create empty mask
            mask = np.zeros_like(gray)

            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > 100:  # Minimum area threshold
                    # Fill the contour
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

                    # Apply morphological operations to clean up the mask
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            return mask

        except Exception as e:
            logging.error(f"Mask creation failed: {str(e)}")
            return None

    def apply_mask_to_depth(self, depth_map, mask):
        """Apply binary mask to depth map for accurate volume calculation."""
        if depth_map is None or mask is None:
            return None

        try:
            # Ensure mask and depth map have same dimensions
            if depth_map.shape != mask.shape:
                mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))

            # Convert mask to binary
            binary_mask = (mask > 0).astype(np.uint8)

            # Apply mask to depth map
            masked_depth = depth_map.copy()
            masked_depth[binary_mask == 0] = 0

            return masked_depth

        except Exception as e:
            logging.error(f"Depth map masking failed: {str(e)}")
            return None
