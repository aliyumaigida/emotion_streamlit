
import os
import mediapipe as mp
import numpy as np
from PIL import Image


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Face Landmarker model path
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_landmarker.task"
)


# Check that the model exists
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Face Landmarker model not found: {MODEL_PATH}"
    )


# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Configure landmark detection
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)


# Create landmarker
landmarker = FaceLandmarker.create_from_options(
    options
)


def detect_landmarks(image):
    """
    Detect facial landmarks in a PIL image.

    Returns:
        A list of dictionaries containing normalized
        x, y, z coordinates, or None if no face is found.
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert image to NumPy array
    image_np = np.asarray(image)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    # Detect landmarks
    result = landmarker.detect(mp_image)

    # Return None if no face is detected
    if not result.face_landmarks:
        return None

    # Extract landmarks for the first detected face
    landmarks = []

    for landmark in result.face_landmarks[0]:
        landmarks.append({
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z)
        })

    return landmarks
