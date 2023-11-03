import os
import random

import torch
import torch.nn.functional as F
from mish_cuda import MishCuda
from model.ab import AccuracyBoosterPlusBlock
from model.resnet import custom_resnet18, custom_resnet50
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np
import streamlit as st


# CONSTANTS
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

st.set_page_config(page_title="Demo Klasifikasi Ekspresi Wajah")
st.title("Demo Klasifikasi Ekspresi Wajah")

classes = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

model = custom_resnet50(
    activation_layer=MishCuda(),
    output_block={
        "class": AccuracyBoosterPlusBlock,
        "params": {},
    },
    num_classes=len(classes),
)
model.load_state_dict(torch.load("output/best-result.pt"))


def capture_face(image_input):
    gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
    )

    return faces


def predict(model, test_data):
    model.cuda()
    model.eval()

    test_data = test_data.cuda()
    outputs = model(test_data)
    _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    picture_from_camera = st.camera_input('Ambil gambar')
    picture_file = st.file_uploader('Atau unggah gambar', type=['jpeg', 'jpg', 'png'])

    while True:
        if picture_from_camera is not None or picture_file is not None:
            st.text('Gambar berhasil ditangkap')
            break

    input_image = Image.open(picture_from_camera if picture_from_camera is not None else picture_file).convert('RGB')
    input_image_np = np.asarray(input_image)
    st.image(input_image, caption='Gambar masukan', width=720)
    cropped_faces = [input_image_np[y : y + h, x : x + w] for (x, y, w, h) in capture_face(input_image_np)]

    if cropped_faces:
        st.text('Gambar wajah yang terdeteksi')
        st.image(image=cropped_faces, width=120)
        predict_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        input_image = torch.stack(
            [predict_transform(Image.fromarray(cf)) for cf in cropped_faces]
        )

        result = predict(model, input_image)
        st.text('Hasil klasifikasi')
        st.image(image=cropped_faces, caption=[classes[res] for res in result], width=120)
    else:
        st.text('Wajah tidak terdeteksi')
