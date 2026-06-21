import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Prediction")

file_path = hf_hub_download(
    repo_id="mshahban/tourism-package-model",
    filename="best_tourism_package_model_v1.joblib"
)

model = joblib.load(file_path)

st.success("Model-2 loaded successfully with joblib")
st.write(type(model))
