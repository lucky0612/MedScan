import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from PIL import Image, ImageOps
from keras.models import load_model

def predict_segmentation_liver(img_path, model, threshold=0.5):
    im_width = 256
    im_height = 256
    img = load_img(img_path, color_mode='grayscale')
    x_img = img_to_array(img)
    x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
    x_img = x_img / 255.0
    x_img = np.expand_dims(x_img, axis=0)
    predictions = model.predict(x_img)
    binary_predictions = (predictions > threshold).astype(np.float32)

    return x_img, binary_predictions

# Load your trained model
model = load_model('/Users/laravi/Downloads/healthcare/Liver_Model.hdf5')

# Streamlit app
st.title('Liver Tumor Segmentation')

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg'])

if uploaded_file is not None:
    with st.spinner('Processing...'):
        x_img, binary_predictions = predict_segmentation_liver(uploaded_file, model)

    col1, col2, col3 = st.columns(3)

    # Plot the original image
    with col1:
        st.image(x_img.squeeze(), caption='Original Image', use_column_width=True)

    # Plot the predicted segmentation mask
    with col2:
        st.image(binary_predictions.squeeze(), caption='Predicted Segmentation Mask', use_column_width=True)

    # Display bounding box image
    with col3:
        # Threshold the prediction to obtain a binary mask
        threshold = 0.5  # Adjust threshold as needed
        binary_mask = binary_predictions.squeeze() > threshold

        # Perform connected component analysis
        labeled_mask = label(binary_mask)

        # Get region properties
        regions = regionprops(labeled_mask)

        # Create a figure with subplots
        fig, ax = plt.subplots()
        ax.imshow(binary_predictions.squeeze(), cmap='gray')

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
