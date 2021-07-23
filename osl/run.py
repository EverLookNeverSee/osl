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


# Instantiating model and passing its configuration parameters
model = model.MaskRCNN(
    mode="inference",
    config=MainConfig(),
    model_dir="./"
)
