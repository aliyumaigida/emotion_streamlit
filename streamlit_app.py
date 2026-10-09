import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import av
from detector.face_landmarks import detect_landmarks
from detector.face_detector import detect_face
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="EmotionVision",
    page_icon="😊",
    layout="centered"
)


# ==========================================
# EMOTION CLASSES
# ==========================================

CLASS_NAMES = [
    "Angry",
    "Fear",
    "Happy",
    "Sad",
    "Surprise"
]


# ==========================================
# MODEL PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_model",
    "emotion_model.keras"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_emotion(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index]
        * 100
    )

    probabilities = {}

    for i, emotion in enumerate(
        CLASS_NAMES
    ):

        probabilities[emotion] = (
            float(predictions[0][i]) * 100
        )

    return (
        predicted_emotion,
        confidence,
        probabilities
    )


# ==========================================
# LIVE WEBCAM PROCESSOR
# ==========================================

class EmotionVideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        face, bbox = detect_face(pil_image)

        if face is not None and bbox is not None:
            x, y, width, height = bbox

            # Draw face bounding box
            cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            # Predict emotion
            emotion, confidence, probabilities = predict_emotion(face)

            # Draw facial landmarks
            landmarks = detect_landmarks(pil_image)

            if landmarks:
                for landmark in landmarks:
                    landmark_x = int(landmark["x"] * img.shape[1])
                    landmark_y = int(landmark["y"] * img.shape[0])

                    cv2.circle(
                        img,
                        (landmark_x, landmark_y),
                        1,
                        (0, 255, 255),
                        -1
                    )

            # Draw emotion and confidence
            cv2.putText(
                img,
                f"Emotion: {emotion}",
                (x, max(y - 40, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f"Confidence: {confidence:.2f}%",
                (x, max(y - 10, 55)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # PROBABILITY BAR PANEL
            panel_x = 10
            panel_y = 10
            panel_width = min(340, img.shape[1] - 20)
            bar_width = panel_width - 30
            bar_height = 12
            row_height = 42

            # Handle probabilities expressed as fractions or percentages
            values = [float(v) for v in probabilities.values()]
            if values and max(values) <= 1.0:
                scale = 100.0
            else:
                scale = 1.0

            # Draw dark panel background
            panel_height = 38 + len(probabilities) * row_height

            cv2.rectangle(
                img,
                (panel_x, panel_y),
                (panel_x + panel_width, min(panel_y + panel_height, img.shape[0] - 1)),
                (35, 35, 35),
                -1
            )

            cv2.putText(
                img,
                "EMOTION PROBABILITIES",
                (panel_x + 10, panel_y + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            bar_y = panel_y + 42

            for label, probability in probabilities.items():
                probability = max(
                    0.0,
                    min(float(probability) * scale, 100.0)
                )

                # Emotion label
                cv2.putText(
                    img,
                    f"{label}: {probability:.1f}%",
                    (panel_x + 10, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1
                )
            

                # Gray bar background
                cv2.rectangle(
                    img,
                    (panel_x + 10, bar_y + 15),
                    (panel_x + 10 + bar_width, bar_y + 15 + bar_height),
                    (100, 100, 100),
                    -1
                )

                # Green probability bar
                filled_width = int(bar_width * probability / 100.0)

                if filled_width > 0:
                    cv2.rectangle(
                        img,
                        (panel_x + 10, bar_y + 15),
                        (panel_x + 10 + filled_width, bar_y + 15 + bar_height),
                        (0, 255, 0),
                        -1
                    )

                bar_y += row_height

        else:
            cv2.putText(
                img,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

tab1, tab2 = st.tabs([
    "📷 Image Upload",
    "🎥 Live Webcam"
])
# ==========================================
# IMAGE UPLOAD
# ==========================================

with tab2:

    st.subheader(
        "🎥 Real-Time Emotion Detection"
    )

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

# ==========================================
# LIVE CAMERA
# ==========================================


with tab2:

    st.subheader("Live Emotion Detection")

    st.write(
        "Allow camera access to start "
        "real-time emotion detection."
    )

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        st.image(
            image,
            caption="Camera Image",
            use_container_width=True
        )

        emotion, confidence, probabilities = (
            predict_emotion(image)
        )

        st.subheader("Prediction")

        st.markdown(
            f"""
            <h3 style="color: green;">
                Emotion: {emotion}
            </h3>
            <h4 style="color: green;">
                Confidence: {confidence:.2f}%
            </h4>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Emotion Probabilities")


        # Display emotion probabilities with bars
        y_position = 100

        for label, probability in probabilities.items():
     

            probability = max(
                0,
                min(float(probability), 100)
            )

            # Emotion name
            cv2.putText(
                img,
                str(label),
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Percentage
            cv2.putText(
                img,
                f"{probability:.2f}%",
                (150, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            # Background bar
            bar_x = 250
            bar_y = y_position - 15
            bar_width = 250
            bar_height = 15

            cv2.rectangle(
                img,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (80, 80, 80),
                -1
            )

            # Green bar based on probability
            filled_width = int(
                bar_width * probability / 100
            )

            if filled_width > 0:
                cv2.rectangle(
                    img,
                    (bar_x, bar_y),
                    (bar_x + filled_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1
                )

            y_position += 40