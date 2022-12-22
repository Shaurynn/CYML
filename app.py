import streamlit as st
import os
from processor import preprocess3d, predict
from tensorflow.keras.models import load_model

col1, col2, col3 = st.columns([1, 24, 1])
col2.title("Alzheimer's Disease Detection")
st.image("data/cn_example.png")

with st.sidebar:

    ATLAS = {
        "MNI305": "atlas/tpl-MNI305_T1w.nii.gz",
        "Empty_slot": "atlas/tpl-MNI305_T1w.nii.gz"}

    MODELS = {
        "Inception": "models/inception_model1.h5",
        "Empty_slot": "models/inception_model1.h5"}

    st.header("Upload subject MRI image:")
    image = st.file_uploader("Choose an image", type=".nii", key="image")

    st.header("Select atlas and model:")
    atlas_key = st.selectbox("Choose a referencece image", [i for i in ATLAS.keys()])
    atlas_value = ATLAS.get(atlas_key)
    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
    model_value = MODELS.get(model_key)

    #st.cache(allow_output_mutation=True)
    def detect_AD():
        preprocess3d(os.path.join("./input",image.name), atlas_value)
        predict(f"./output/{image.name}_2d.npy", chosen_model)

    with st.spinner('Loading model'):
        if st.button("Load model"):
            if image is not None and atlas_value is not None and model_value is not None:
                with open(os.path.join("./input",image.name),"wb") as f:
                    f.write(image.getbuffer())
                chosen_model = load_model(model_value)
                #st.success("Model loaded")
                trigger = st.button('Begin analysis', on_click=detect_AD)
            else:
                st.error("Error: One or more input is/are missing.", icon="🚨")
