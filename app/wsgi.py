"""This script is responsible for starting a Flask application."""

import os
from app.views import app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
