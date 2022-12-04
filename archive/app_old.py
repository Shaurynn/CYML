import streamlit as st
from streamlit_extras.switch_page_button import switch_page
#import os
#from processor import preprocess, predict
#from tensorflow.keras.models import load_model


st.title("Alzheimer's Disease Detection")

col1, col2, col3= st.columns([1, 3, 1])

go_next = col3.button("Begin")
if go_next:
    switch_page("Start")

#with st.container():
#    MODELS = {
#        "Inception": "models/inception_model1.h5",
#        "ResNet3D": "models/inception_model1.h5"}
#
#    st.title("Alzheimer's Disease Detection")

#    image = st.file_uploader("Choose an image")
#    atlas = st.file_uploader("Choose an atlas")
#    model_key = st.selectbox("Choose a deep-learning model", [i for i in MODELS.keys()])
#    model_value = MODELS.get(model_key)
#
#    col1, col2, col3= st.columns([1, 3, 1])
#
#    start_test = st.button("Begin analysis")
#    if start_test:
#        if image is not None and atlas is not None and model_value is not None:
#            with open(os.path.join("./input",image.name),"wb") as f:
#                f.write(image.getbuffer())
#            with open(os.path.join("./input", atlas.name),"wb") as a:
#                a.write(atlas.getbuffer())
#            chosen_model = load_model(model_value)
#            switch_page("Results")

    #with st.spinner('Loading model'):
    #    if col3.button("Prepare data"):
    #        if image is not None and atlas is not None and model_value is not None:
    #            with open(os.path.join("./input",image.name),"wb") as f:
    #                f.write(image.getbuffer())
    #            with open(os.path.join("./input", atlas.name),"wb") as a:
    #                a.write(atlas.getbuffer())
    #            chosen_model = load_model(model_value)
    #        st.success("Ready")
