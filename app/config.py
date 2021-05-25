"""This module contains configurations for various environments."""

import os


basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    MODEL_NAME = 'resnet50'
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'resnet50_lr1e-3_bs32.pth')
    CATEGORIES_PATH = os.path.join(os.getcwd(), 'data', 'categories.json')


class ProdConfig(Config):
    DEBUG = False
    TESTING = False


class DevConfig(Config):
    DEBUG = True
    TESTING = True
