
import os

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detector.face_detector import detect_face

st.set_page_config(page_title="EmotionVision", page_icon="😊")
st.title("EmotionVision — Deployment Test")

CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ai_model",
    "emotion_model.keras"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    st.success("Emotion model loaded successfully.")
except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
    st.stop()


def predict_emotion(face):
    face = face.convert("RGB").resize((224, 224))
    image_array = np.asarray(face, dtype=np.float32)[None, ...]

    # Run one inference directly instead of using model.predict()
    predictions = model(image_array, training=False).numpy()[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index]) * 100

    return CLASS_NAMES[index], confidence


class EmotionTestProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            face, bbox = detect_face(pil_image)

            if face is not None and bbox is not None:
                x, y, w, h = map(int, bbox)

                emotion, confidence = predict_emotion(face)

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"{emotion}: {confidence:.1f}%",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    image,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        except Exception as e:
            print(f"Prediction error: {type(e).__name__}: {e}")
            cv2.putText(
                image,
                "Prediction failed - check logs",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="emotionvision-test",
    video_processor_factory=EmotionTestProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},
        "audio": False
    },
    async_processing=True
)
