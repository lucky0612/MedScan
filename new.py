import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from PIL import Image, ImageOps
from keras.models import load_model
import base64
from io import BytesIO

# Function to load and preprocess the image
def load_and_preprocess(img_path, model, threshold=0.5):
    size = (256, 256)
    image = Image.open(img_path)
    image = ImageOps.fit(image, size)
    x_img = img_to_array(image)
    x_img = resize(x_img, (256, 256, 1), mode='constant', preserve_range=True)
    x_img /= 255.0
    x_img = np.expand_dims(x_img, axis=0)
    predictions = model.predict(x_img)
    binary_predictions = (predictions > threshold).astype(np.float32)
    return binary_predictions.squeeze()

# Function to predict and display segmentation
def predict_and_display_segmentation(model_path, title, index):
    st.header(f'{title} - {index}')
    uploaded_file = st.file_uploader(f"Upload an image {index}", type=['png', 'jpg'], key=f"file_{index}")
    if uploaded_file is not None:
        model = load_model(model_path)
        with st.spinner('Processing...'):
            result = load_and_preprocess(uploaded_file, model)

        threshold = 0.5  # Adjust threshold as needed
        binary_mask = result > threshold
        labeled_mask = label(binary_mask)
        regions = regionprops(labeled_mask)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        original_image = Image.open(uploaded_file)
        ax[0].imshow(original_image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Plot segmented image
        ax[1].imshow(result, cmap='gray')
        ax[1].set_title('Segmented Image')
        ax[1].axis('off')

        # Plot bounding box image
        ax[2].imshow(result, cmap='gray')
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            ax[2].add_patch(plt.Rectangle((minc, minr), width, height, fill=False, edgecolor='red', linewidth=2))
        ax[2].set_title('Bounding Box')
        ax[2].axis('off')

        st.pyplot(fig)

# Function to set background image with blur
def set_background():
    background_image = Image.open("/Users/laravi/Downloads/healthcare/background.png")
    img_io = BytesIO()
    background_image.save(img_io, format='PNG')
    img_io.seek(0)
    img_data = base64.b64encode(img_io.getvalue()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{img_data}');
            background-size: cover;
            background-attachment: fixed;
            backdrop-filter: blur(3600px);
            color: black;
            }}
            h1, h2, h3, h4, h5, h6, p, span, label, div, a, li, strong, b {{
            color: black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit app
def main():
    # Set background
    set_background()

    # Title
    st.title('MedScan')

    # Display segmentation for different models
    predict_and_display_segmentation('/Users/laravi/Downloads/healthcare/CC_Model.hdf5', 'Cardiac Catheterization', 1)
    predict_and_display_segmentation('/Users/laravi/Downloads/healthcare/BT_Model.hdf5', 'Brain Tumor Segmentation', 2)
    predict_and_display_segmentation('/Users/laravi/Downloads/healthcare/Liver_Model.hdf5', 'Liver Tumor Segmentation', 3)
    predict_and_display_segmentation('/Users/laravi/Downloads/healthcare/Viral_Pneumonia.hdf5', 'Viral Pneumonia Segmentation', 4)
    predict_and_display_segmentation('ALL_MODELS/COVID.hdf5', 'COVID Segmentation', 5)

if __name__ == "__main__":
    main()
