import os
from flask import Flask
from .config import config_by_name # Relative import for config within the package

# Create the Flask application instance
# Explicitly set template_folder based on your structure
# Explicitly set static_folder based on convention and config.py usage
app = Flask(__name__,
            template_folder='Templates/HTML',
            static_folder='static')

# Load configuration from config.py
# Use environment variable FLASK_CONFIG or default to 'development'
config_name = os.getenv('FLASK_CONFIG', 'development')
app.config.from_object(config_by_name[config_name])

# Ensure the upload folder exists after the config is loaded
# This is better placed here than in config.py as it uses app.config
upload_folder = app.config.get('UPLOAD_FOLDER')
if upload_folder:
    try:
        os.makedirs(upload_folder, exist_ok=True) # exist_ok=True prevents error if folder exists
        # Optional: Log folder creation if needed, requires app context or standard logging setup
        # app.logger.info(f"Ensured upload folder exists: {upload_folder}")
    except OSError as e:
        # Log the error if folder creation fails
        app.logger.error(f"Could not create upload folder {upload_folder}: {e}")
else:
    app.logger.warning("UPLOAD_FOLDER is not configured.")


# Import routes after the app object is created to avoid circular imports
from App import routes

# Optional: Add logging configuration here if needed
# Example:
# if not app.debug:
#     import logging
#     from logging.handlers import RotatingFileHandler
#     if not os.path.exists('logs'):
#         os.mkdir('logs')
#     file_handler = RotatingFileHandler('logs/rdc_concrete.log', maxBytes=10240, backupCount=10)
#     file_handler.setFormatter(logging.Formatter(
#         '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
#     file_handler.setLevel(logging.INFO)
#     app.logger.addHandler(file_handler)
#     app.logger.setLevel(logging.INFO)
#     app.logger.info('RDC Concrete startup')