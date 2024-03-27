import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten,LeakyReLU,BatchNormalization
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers import Conv2D, Conv2DTranspose,DepthwiseConv2D
from keras.layers import MaxPooling2D, GlobalMaxPool2D, Concatenate
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tqdm import tqdm
import streamlit as st
from PIL import Image,ImageOps
from skimage.measure import label, regionprops

im_width = 256
im_height = 256

# Load the pre-trained model
model = load_model('/Users/laravi/Downloads/healthcare/CC_Model.hdf5')

st.header('cardiac_catheterization')

file = st.file_uploader("Please upload a cardiac_catheterization image", type=['jpg', 'png'])

def load_and_preprocess(img_path):
    size=(256,256)
    image=ImageOps.fit(img_path,size)
    x_img = img_to_array(image)
    x_img = resize(x_img, (im_height, im_width), mode='constant', preserve_range=True)
    x_img /= 255.0
    x_img = np.expand_dims(x_img, axis=0)
    prediction = model.predict(x_img)
    return prediction.squeeze()

if file is None:
    st.text('Please upload an image file')
else:
    image = Image.open(file)
    result = load_and_preprocess(image)
    col1, col2, col3 = st.columns(3)
    
    # Display original image
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
    
    # Display processed image
    with col2:
        st.image(result, caption='Processed Image', use_column_width=True)
    
    # Display bounding box image
    with col3:
        # Threshold the prediction to obtain a binary mask
        threshold = 0.5  # Adjust threshold as needed
        binary_mask = result > threshold

        # Perform connected component analysis
        labeled_mask = label(binary_mask)

        # Get region properties
        regions = regionprops(labeled_mask)

        # Create a figure with subplots
        fig, ax = plt.subplots()
        ax.imshow(result, cmap='gray')

        # Draw bounding boxes around regions
        for region in regions:
            # Get bounding box coordinates
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            area = region.area

            # Draw bounding box
            rect = plt.Rectangle((minc, minr), width, height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        plt.axis('off')
        st.pyplot(fig)
