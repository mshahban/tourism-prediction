import streamlit as st
from huggingface_hub import hf_hub_download

st.title("Tourism Prediction")

file_path = hf_hub_download(
    repo_id="mshahban/tourism-package-model",
    filename="best_tourism_package_model_v1.joblib"
)

st.success("Model downloaded successfully")
st.write(file_path)
