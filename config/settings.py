# filepath: python-server-project/config/settings.py

# Configuration settings for the application
import os

class Config:
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///site.db')
    STATIC_FOLDER = 'static'
    TEMPLATES_FOLDER = 'templates'