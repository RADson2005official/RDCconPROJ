import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from .config import config_by_name # Relative import for config

# Determine the absolute path to the App directory
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir) # Project root directory

# Create the Flask application instance
# Explicitly set template_folder and static_folder relative to this file's directory (App)
app = Flask(__name__,
            template_folder=os.path.join(basedir, 'Templates', 'HTML'),
            static_folder=os.path.join(basedir, 'static'))

# Load configuration based on FLASK_CONFIG environment variable
config_name = os.getenv('FLASK_CONFIG', 'development')
try:
    app.config.from_object(config_by_name[config_name])
    print(f" * Loading configuration: '{config_name}'")
except KeyError:
    print(f" ! Warning: Invalid FLASK_CONFIG value '{config_name}'. Using 'development'.")
    config_name = 'development'
    app.config.from_object(config_by_name[config_name])

# --- Logging Setup ---
if not app.debug and not app.testing:
    log_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            print(" ! Error: Could not create logs directory.")

    log_file = os.path.join(log_dir, 'rdc_concrete.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('RDC Concrete Application startup')
else:
    # Use basic logging in debug mode (Werkzeug often adds handlers too)
    logging.basicConfig(level=logging.DEBUG)
    app.logger.info('RDC Concrete Application startup (Debug Mode)')


# --- Ensure Essential Folders Exist ---
# UPLOAD_FOLDER from config might be relative to project root
upload_folder_config = app.config.get('UPLOAD_FOLDER')
if upload_folder_config:
    # Construct absolute path if relative
    if not os.path.isabs(upload_folder_config):
        upload_folder_abs = os.path.join(project_root, upload_folder_config)
    else:
        upload_folder_abs = upload_folder_config

    try:
        os.makedirs(upload_folder_abs, exist_ok=True)
        app.logger.info(f"Ensured upload folder exists: {upload_folder_abs}")
        # Update config with absolute path for consistency if needed elsewhere
        app.config['UPLOAD_FOLDER_ABS'] = upload_folder_abs
    except OSError as e:
        app.logger.error(f"Could not create upload folder {upload_folder_abs}: {e}")
else:
    app.logger.warning("UPLOAD_FOLDER is not configured in config.py.")

# Results folder (relative to static folder)
results_folder = os.path.join(app.static_folder, 'results')
try:
    os.makedirs(results_folder, exist_ok=True)
    app.logger.info(f"Ensured results folder exists: {results_folder}")
except OSError as e:
    app.logger.error(f"Could not create results folder {results_folder}: {e}")


# Import routes AFTER app and config are defined to avoid circular imports
from App import routes
app.logger.info("Application routes imported.")