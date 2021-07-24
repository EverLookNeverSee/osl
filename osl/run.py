"""
    This file contains:
        - Loading and configuring Mask-Rcnn pre-trained model
        - Building web app using streamlit package
"""

import numpy as np
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

# Loading model weights
weights = "weights/mask_rcnn_coco.h5"
model.load_weights(filepath=weights, by_name=True)

# Model class names
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
               'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# App title and short description
st.title("OSL")
st.subheader("Object Segmenting and Labeling")

try:
    image = st.file_uploader(label="Upload your image:")
    if image is not None:
        st.success("Image uploaded successfully")
        image = Image.open(image)
        st.image(image, caption="Test Image")
        image = np.array(image)
        results = model.detect([image], verbose=0)
        result = results[0]
        detected_labels = [CLASS_NAMES[item] for item in result.get("class_ids")]
        masked_image = visualize.display_instances(image=image,
                                                   boxes=result['rois'],
                                                   masks=result['masks'],
                                                   class_ids=result['class_ids'],
                                                   class_names=CLASS_NAMES,
                                                   scores=result['scores'])
        st.image(masked_image, caption="Masked Image")
        for item in detected_labels:
            st.success(item)

except ValueError:
    st.error("Upload failed!")
