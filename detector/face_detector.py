
import os
import numpy as np
import mediapipe as mp
from PIL import Image

# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# MediaPipe face detection model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "blaze_face_short_range.tflite"
)

# Check that the model file exists
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Face detection model not found: {MODEL_PATH}"
    )

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure face detector
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.5
)

# Create detector
detector = FaceDetector.create_from_options(options)


def detect_face(image):
    """
    Detect the first face in a PIL image.

    Returns:
        face: Cropped PIL image, or None if no face is detected.
        bbox: (x, y, width, height), or None if no face is detected.
    """

    # Ensure RGB format
    image = image.convert("RGB")

    # Convert to NumPy array
    image_np = np.asarray(image)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_np
    )

    # Detect face
    result = detector.detect(mp_image)

    if not result.detections:
        return None, None

    # Select the first detected face
    detection = result.detections[0]
    bbox = detection.bounding_box

    # Get bounding-box coordinates
    x = int(bbox.origin_x)
    y = int(bbox.origin_y)
    width = int(bbox.width)
    height = int(bbox.height)

    # Clamp coordinates to image boundaries
    image_width, image_height = image.size

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + width)
    y2 = min(image_height, y + height)

    # Reject invalid bounding boxes
    if x2 <= x1 or y2 <= y1:
        return None, None

    # Crop the face
    face = image.crop((x1, y1, x2, y2))

    # Return the actual crop coordinates
    corrected_bbox = (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )

    return face, corrected_bbox
