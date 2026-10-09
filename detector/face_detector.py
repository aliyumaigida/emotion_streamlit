import os
import numpy as np
import mediapipe as mp
from PIL import Image


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "blaze_face_short_range.tflite"
)


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE
)


detector = FaceDetector.create_from_options(options)


def detect_face(image):

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # Convert to NumPy
    image_np = np.array(image)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    # Detect face
    result = detector.detect(mp_image)

    if not result.detections:
        return None, None

    # Use the first detected face
    detection = result.detections[0]

    bbox = detection.bounding_box

    x = bbox.origin_x
    y = bbox.origin_y
    width = bbox.width
    height = bbox.height

    # Crop the face
    face = image.crop(
        (x, y, x + width, y + height)
    )

    return face, (x, y, width, height)