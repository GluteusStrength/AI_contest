import torch
import numpy as np
import streamlit as st
from torchvision.transforms import transforms


@st.cache_data()
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt', force_reload=True)


def predict(img):
    DEVICE = torch.device('mps:0' if torch.backends.mps.is_available(
    ) else 'cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_model()
    model.to(DEVICE)

    results = model(img)

    return results
