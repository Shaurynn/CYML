import streamlit as st
import os
from ants import image_read, plot
from processor import preprocess, predict
from tensorflow.keras.models import load_model


with st.container():
    MODELS = {
        "Inception": "models/inception_model1.h5",
        "ResNet3D": "models/inception_model1.h5"}

    st.title("Alzheimer's Disease Detection")

    image = st.file_uploader("Choose an image")
    atlas = st.file_uploader("Choose an atlas")
    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
    model_value = MODELS.get(model_key)

    col1, col2, col3= st.columns([1, 3, 1])

    with st.spinner('Loading model'):
        if col3.button("Prepare data"):
            if image is not None and atlas is not None and model_value is not None:
                with open(os.path.join("./",image.name),"wb") as f:
                    f.write(image.getbuffer())
                    plot(image_read(os.path.join("./",image.name)))
                with open(os.path.join("./", atlas.name),"wb") as a:
                    a.write(atlas.getbuffer())
                chosen_model = load_model(model_value)
            st.success("Ready")

    st.cache(allow_output_mutation=True)
    def detect_AD():
        preprocess(os.path.join("./",image.name),os.path.join("./", atlas.name))
        predict(f"./{image.name}_2d.npy", chosen_model)

trigger = col3.button('Predict', on_click=detect_AD)
