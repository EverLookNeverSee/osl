"""
    This file contains:
        - Loading and configuring Mask-Rcnn pre-trained model
        - Building web app using streamlit package
"""

from PIL import Image
import streamlit as st
from mrcnn import model, config, visualize
