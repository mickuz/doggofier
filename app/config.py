import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'resnet50.pth')
    CATEGORIES_PATH = os.path.join(os.getcwd(), 'data', 'categories.json')


class ProdConfig(Config):
    DEBUG = False
    TESTING = False


class DevConfig(Config):
    DEBUG = True
    TESTING = True
