import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os
from processor import preprocess, predict
from tensorflow.keras.models import load_model
from pages import Start

with st.container():
    MODELS = {
        "Inception": "models/inception_model1.h5",
        "ResNet3D": "models/inception_model1.h5"}

    st.header("Please upload subject MRI image and reference atlas.")
    image = st.file_uploader("Choose an image", type=".nii", key="image")
    atlas = st.file_uploader("Choose an atlas", type=".nii.gz", key="atlas")

    st.header("Please select a pre-train model for analysis.")
    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
    model_value = MODELS.get(model_key)

    col1, col2, col3= st.columns([1, 3, 1])

    st.cache(allow_output_mutation=True)
    def detect_AD():
        preprocess(os.path.join("./input",image.name),os.path.join("./input", atlas.name))
        predict(f"./output/{image.name}_2d.npy", chosen_model)

    with st.spinner('Loading model'):
        if col3.button("Load model"):
            if image is not None and atlas is not None and model_value is not None:
                with open(os.path.join("./input",image.name),"wb") as f:
                    f.write(image.getbuffer())
                with open(os.path.join("./input", atlas.name),"wb") as a:
                    a.write(atlas.getbuffer())
                chosen_model = load_model(model_value)
                st.success("Model loaded")
                trigger = col3.button('Being analysis', on_click=detect_AD)
            else:
                st.error("Error: Subject image and/or reference atlas missing.", icon="🚨")
