import os

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detector.face_detector import detect_face
from detector.face_landmarks import detect_landmarks

st.set_page_config(page_title="EmotionVision", page_icon="😊")
st.title("EmotionVision — Real-Time Emotion Recognition")

CLASS_NAMES = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ai_model",
    "emotion_model.keras"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()
st.success("Emotion model loaded successfully.")

# These are shared display areas outside the webcam.
st.subheader("Emotion probabilities")
probability_area = st.empty()


def predict_emotion(face):
    face = face.convert("RGB").resize((224, 224))
    image_array = np.asarray(face, dtype=np.float32)[None, ...]

    predictions = model(image_array, training=False).numpy()[0]
    index = int(np.argmax(predictions))

    return CLASS_NAMES[index], float(predictions[index]) * 100, predictions


class EmotionTestProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            face, bbox = detect_face(pil_image)

            if face is not None and bbox is not None:
                x, y, w, h = map(int, bbox)

                emotion, confidence, probabilities = predict_emotion(face)

                cv2.rectangle(
                    image, (x, y), (x + w, y + h), (0, 255, 0), 2
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

                # Draw facial landmarks.
                # This assumes detect_landmarks returns normalized
                # (x, y) coordinates between 0 and 1.
                try:
                    landmarks = detect_landmarks(pil_image)

                    if landmarks is not None:
                        for point in landmarks:
                            if hasattr(point, "x") and hasattr(point, "y"):
                                px = int(point.x * image.shape[1])
                                py = int(point.y * image.shape[0])
                            else:
                                px = int(point[0] * image.shape[1])
                                py = int(point[1] * image.shape[0])

                            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                                cv2.circle(image, (px, py), 1, (0, 255, 255), -1)

                except Exception as landmark_error:
                    print("Landmark error:", repr(landmark_error))

                # Store probabilities for the Streamlit interface.
                self.latest_probabilities = {
                    name: float(value) * 100
                    for name, value in zip(CLASS_NAMES, probabilities)
                }

            else:
                cv2.putText(
                    image, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

        except Exception as e:
            print(f"Prediction error: {type(e).__name__}: {e}")

        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="emotionvision-test",
    video_processor_factory=EmotionTestProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)

# Display the latest probability scores.
if ctx.video_processor:
    latest = getattr(ctx.video_processor, "latest_probabilities", None)

    if latest:
        with probability_area.container():
            for name, score in latest.items():
                st.write(f"**{name}: {score:.1f}%**")
                st.progress(min(max(int(round(score)), 0), 100))
