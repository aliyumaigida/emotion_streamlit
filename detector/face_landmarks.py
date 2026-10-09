import os
import mediapipe as mp
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(
    options
)


def detect_landmarks(image):

    image = image.convert("RGB")

    image_np = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = []

    for landmark in result.face_landmarks[0]:

        landmarks.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z
        })

    return landmarks