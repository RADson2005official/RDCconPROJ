from flask import render_template, request, jsonify, redirect, url_for, flash, current_app, send_from_directory
import os
import time
import numpy as np
from werkzeug.utils import secure_filename
from App import app  # Import the app instance
import logging  # Import logging

# Use relative imports for Utils within the same package
from .Utils.ImageLoader import ImageLoader
from .Utils.ImageProcessor import ImageProcessor
from .Utils.segmentation import perform_segmentation, visualize_segmentation
from .Utils.Calculation import Calculation
from .Utils.Visualizer import Visualizer

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('main.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/process_uploaded_images', methods=['POST'])
def process_uploaded_images_route():
    """
    Handles processing of uploaded top and front view images via POST request.
    """
    start_time = time.time()
    current_app.logger.info("Received request for /process_uploaded_images")
    calculation_instance = Calculation()  # Instantiate per request

    if 'topImageFile' not in request.files or 'frontImageFile' not in request.files:
        return jsonify({'error': "Missing 'topImageFile' or 'frontImageFile' in request."}), 400

    top_file = request.files['topImageFile']
    front_file = request.files['frontImageFile']

    if top_file.filename == '' or front_file.filename == '':
        return jsonify({'error': "No file selected for top or front view."}), 400

    if top_file and allowed_file(top_file.filename) and front_file and allowed_file(front_file.filename):
        try:
            # --- Setup Folders ---
            upload_folder = app.config.get('UPLOAD_FOLDER_ABS')
            results_folder = os.path.join(current_app.static_folder, 'results')
            if not upload_folder or not os.path.isdir(upload_folder):
                msg = f"Upload folder is not configured correctly or does not exist: {upload_folder}"
                current_app.logger.error(msg)
                return jsonify({'error': msg}), 500
            os.makedirs(results_folder, exist_ok=True)

            # --- Save Original Files ---
            ts = int(time.time())
            top_filename_orig = secure_filename(f"top_{ts}_orig_{top_file.filename}")
            front_filename_orig = secure_filename(f"front_{ts}_orig_{front_file.filename}")
            top_filepath_orig = os.path.join(upload_folder, top_filename_orig)
            front_filepath_orig = os.path.join(upload_folder, front_filename_orig)
            top_file.save(top_filepath_orig)
            front_file.save(front_filepath_orig)
            current_app.logger.info(f"Saved original top image: {top_filepath_orig}")
            current_app.logger.info(f"Saved original front image: {front_filepath_orig}")

            # --- Processing Pipeline ---
            # 1. Load Originals
            loader = ImageLoader(image_dir=upload_folder)
            top_image_orig = loader.load_image(top_filepath_orig)
            front_image_orig = loader.load_image(front_filepath_orig)
            if top_image_orig is None or front_image_orig is None:
                raise ValueError("Failed to load one or both saved original images")

            # 2. Preprocess
            processor = ImageProcessor()
            processed_top = processor.preprocess_image(top_image_orig)
            processed_front = processor.preprocess_image(front_image_orig)
            if processed_top is None or processed_front is None:
                raise ValueError("Failed to preprocess one or both images")

            # 3. Segment
            top_seg_data = perform_segmentation(processed_top)
            front_seg_data = perform_segmentation(processed_front)
            if top_seg_data is None or front_seg_data is None:
                raise ValueError("Segmentation failed for one or both images")

            # 4. Visualize Segmentation
            top_seg_viz_filename = f"top_{ts}_seg.png"
            front_seg_viz_filename = f"front_{ts}_seg.png"
            top_seg_viz_path = os.path.join(results_folder, top_seg_viz_filename)
            front_seg_viz_path = os.path.join(results_folder, front_seg_viz_filename)
            viz_top_success = visualize_segmentation(top_image_orig, top_seg_data, top_seg_viz_path)
            viz_front_success = visualize_segmentation(front_image_orig, front_seg_data, front_seg_viz_path)
            top_seg_url = url_for('static', filename=f'results/{top_seg_viz_filename}') if viz_top_success else None
            front_seg_url = url_for('static', filename=f'results/{front_seg_viz_filename}') if viz_front_success else None

            # 5. Calculate Volume, Mass, Density using the Class instance
            material_type = request.form.get('materialType', 'Gravel')  # Default to Gravel
            condition = request.form.get('materialCondition', 'Loose')  # Default to Loose
            current_app.logger.info(f"Using Material: {material_type}, Condition: {condition} for calculation.")

            calculation_result = calculation_instance.calculate_from_segmentation(
                top_seg_data, front_seg_data, material_type, condition
            )

            if calculation_result is None:
                raise ValueError("Calculation failed (returned None)")

            # 6. Visualize 3D
            viz_3d_relative_path = None
            try:
                visualizer = Visualizer()
                visualizer.plot_cube(
                    corners_3d=calculation_result.get('points'),
                    volume=calculation_result.get('volume'),
                    density=calculation_result.get('density'),
                    mass=calculation_result.get('mass')  # Pass mass in KG
                )
                viz_3d_filename = f"visualization_3d_{ts}.png"
                viz_3d_abs_path = os.path.join(results_folder, viz_3d_filename)
                if visualizer.save_figure(viz_3d_abs_path):
                    viz_3d_relative_path = os.path.join('results', viz_3d_filename).replace('\\', '/')
                    current_app.logger.info(f"3D Visualization saved to {viz_3d_abs_path}")
                else:
                    current_app.logger.error("Failed to save 3D visualization figure.")
            except Exception as viz_err:
                current_app.logger.error(f"Error during 3D visualization: {viz_err}", exc_info=True)

            # 7. Prepare Response
            processing_time = time.time() - start_time
            current_app.logger.info(f"Processing complete. Time: {processing_time:.2f}s")
            response_data = {
                'status': 'success',
                'volume': calculation_result.get('volume'),
                'mass': calculation_result.get('mass'),  # Mass in KG
                'density': calculation_result.get('density'),  # Density in kg/m³
                'unit_volume': 'm³',
                'unit_mass': 'kg',
                'unit_density': 'kg/m³',
                'processing_time': round(processing_time, 2),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'top_segmentation_url': top_seg_url,
                'front_segmentation_url': front_seg_url,
                'visualization_3d_url': url_for('static', filename=viz_3d_relative_path) if viz_3d_relative_path else None
            }
            return jsonify(response_data)

        except ValueError as ve:
            current_app.logger.error(f"Processing error: {ve}", exc_info=True)
            if "'float' is not iterable" in str(ve):
                error_msg = "Calculation error: Issue processing material density values."
            else:
                error_msg = f'Processing error: {ve}'
            return jsonify({'error': error_msg}), 500
        except Exception as e:
            current_app.logger.error(f"Unexpected error during image processing pipeline: {e}", exc_info=True)
            if isinstance(e, TypeError) and "'float' object is not iterable" in str(e):
                error_msg = "Internal calculation error: Issue processing material density values."
            else:
                error_msg = 'An internal server error occurred during processing.'
            return jsonify({'error': error_msg}), 500
    else:
        return jsonify({'error': f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

@app.route('/api/health')
def health_check():
    """API endpoint for health checks."""
    return jsonify({'status': 'ok'})