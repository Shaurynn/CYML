import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.title("Alzheimer's Disease Detection")

col1, col2, col3= st.columns([1, 3, 1])

go_next = col3.button("Begin")
if go_next:
    switch_page("Start")
