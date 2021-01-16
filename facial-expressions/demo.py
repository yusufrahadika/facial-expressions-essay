import os
import random
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from imgaug import augmenters as iaa
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

st.title("Demo Klasifikasi Wajah")
FRAME_WINDOW = st.image([])

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
model.load_state_dict(torch.load("output/feasible-totem-102.pt"))


def check_webcam():
    webcam_indices = []
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_indices.append(i)
            cap.release()

    return webcam_indices


def capture_face(video_capture):
    # got 3 frames to auto adjust webcam light
    for i in range(3):
        video_capture.read()

    while True:
        ret, frame = video_capture.read()
        
        if frame is None:
            return None, []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        rgb_frame = frame[:, :, ::-1]
        FRAME_WINDOW.image(rgb_frame, width=480)

        return rgb_frame, faces


def predict(model, test_data):
    model.cuda()
    model.eval()

    test_data = test_data.cuda()
    outputs = model(test_data)
    _, predicted = torch.max(outputs, 1)

    return predicted


if __name__ == "__main__":
    tp = st.button("Take a Photo")
    st.button("Restart")

    result_window = st.image([])

    filename = st.text_input("Enter a file path:")

    while True:
        video_capture = cv2.VideoCapture(1)
        frame, faces = capture_face(video_capture)

        if tp or filename:
            break

    if filename:
        try:
            if os.path.isfile(filename):
                cropped_faces = [np.asarray(Image.open(filename))]
        except FileNotFoundError:
            st.error("File not found.")
    else:
        cropped_faces = [frame[y : y + h, x : x + w] for (x, y, w, h) in faces]

    if cropped_faces:
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
        st.image(image=cropped_faces, caption=[classes[res] for res in result])
