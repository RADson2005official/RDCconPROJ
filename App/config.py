import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    # Define the base directory of the application
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) # This should point to 'e:\RDC CONCRETE PROJECT'

    # Define the upload folder relative to the 'App' directory's static folder
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'App', 'static', 'uploads')
    # Define allowed file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    # Optional: Maximum file size (e.g., 16MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Ensure the upload folder exists when the config is loaded
    # Note: This might be better placed in __init__.py or run.py to ensure app context
    # if not os.path.exists(UPLOAD_FOLDER):
    #     os.makedirs(UPLOAD_FOLDER)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    # Add any development-specific settings here


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    # Add any production-specific settings here (e.g., database URI, logging)


# Dictionary to access configurations by name
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}