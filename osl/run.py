"""
    This file contains:
        - Loading and configuring Mask-Rcnn pre-trained model
        - Building web app using streamlit package
"""

from PIL import Image
import streamlit as st
from mrcnn import model, config, visualize


class MainConfig(config.Config):
    """ Model Configuration Class """
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81
