from flask import render_template, request, jsonify, redirect, url_for, flash, current_app
import os
from werkzeug.utils import secure_filename
from App import app
from App.Utils.ImageLoader import ImageLoader
from App.Utils.ImageProcessor import ImageProcessor
from App.Utils.segmentation import perform_segmentation
from App.Utils.Calculation import calculate_damage
from App.Utils.Visualizer import visualize_results

# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page."""
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure the upload folder exists
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Process the image and return results
        return process_image(filepath)

    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_image_route():
    """Process an image with a provided path."""
    data = request.get_json()
    image_path = data.get('image_path')

    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400

    return process_image(image_path)

def process_image(image_path):
    """Process the image through the concrete damage detection pipeline."""
    try:
        # Load the image
        image_loader = ImageLoader()
        image = image_loader.load_image(image_path)

        # Process the image
        processor = ImageProcessor()
        processed_image = processor.preprocess(image)

        # Perform segmentation
        segmentation_result = perform_segmentation(processed_image)

        # Calculate damage metrics
        damage_metrics = calculate_damage(segmentation_result)

        # Generate visualization for the results
        # Ensure the results folder exists
        results_folder = os.path.join(current_app.static_folder, 'results')
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        result_image_path = visualize_results(image, segmentation_result, results_folder)

        # Create relative path for the result image to serve it
        # The base for relative path should be the static folder
        relative_result_path = os.path.relpath(result_image_path, current_app.static_folder)
        # Ensure forward slashes for URL
        relative_result_path = relative_result_path.replace('\\', '/')

        return jsonify({
            'status': 'success',
            'damage_metrics': damage_metrics,
            'result_image': url_for('static', filename=relative_result_path)
        })

    except Exception as e:
        # Log the exception for debugging
        current_app.logger.error(f"Error processing image {image_path}: {e}")
        return jsonify({'error': 'An internal error occurred during image processing.'}), 500

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/api/health')
def health_check():
    """API endpoint for health checks."""
    return jsonify({'status': 'ok'})