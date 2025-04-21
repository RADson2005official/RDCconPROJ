import os
from App import app # Import the app instance created in App/__init__.py

if __name__ == '__main__':
    # Use environment variables or defaults for host, port, debug
    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    try:
        port = int(os.environ.get('FLASK_RUN_PORT', '5000'))
    except ValueError:
        port = 5000

    # Check FLASK_DEBUG env var, default to True if not set or '1'
    # In production, set FLASK_DEBUG=0 or FLASK_CONFIG=production
    is_debug = os.environ.get('FLASK_DEBUG', '1') == '1'

    print(f" * Starting Flask server on http://{host}:{port}")
    print(f" * Debug mode: {'on' if is_debug else 'off'}")
    app.run(host=host, port=port, debug=is_debug)