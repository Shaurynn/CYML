import streamlit as st
import os
from SimpleITK import ReadImage, GetArrayFromImage
from archive.processor_old import preprocess, predict
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from SimpleITK import ReadImage, GetArrayFromImage
import matplotlib.pyplot as plt

with st.container():
    MODELS = {
        "Inception": "models/inception_model1.h5",
        "ResNet3D": "models/inception_model1.h5"}

    st.title("Alzheimer's Disease Detection")

    image = st.file_uploader("Choose an image", type=[".nii"], accept_multiple_files=True)
    atlas = st.file_uploader("Choose an atlas", type=[".nii.gz"])
    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
    model_value = MODELS.get(model_key)

    col1, col2, col3= st.columns([1, 3, 1])

    with st.spinner('Loading model'):
        if col3.button("Prepare data"):
            if image is not None and atlas is not None and model_value is not None:
                for img in image:
                    with open(os.path.join("./input",img.name),"wb") as f:
                        f.write(img.getbuffer())
                with open(os.path.join("./atlas", atlas.name),"wb") as a:
                    a.write(atlas.getbuffer())
                chosen_model = load_model(model_value)
            st.success("Ready")

    st.cache(allow_output_mutation=True)
    def detect_AD():
        atlas_file = os.path.join("./atlas", atlas.name)
        for nii in os.listdir("./input"):
            image_file = os.path.join("./input",nii)
            preprocess(image_file, atlas_file)
            predict(image_file, chosen_model)
            os.remove(image_file)
        os.remove(os.path.join("./atlas", atlas.name))

trigger = col3.button('Predict', on_click=detect_AD)
