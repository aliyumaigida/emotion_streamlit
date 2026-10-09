
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detector.face_detector import detect_face
from detector.face_landmarks import detect_landmarks
import os

st.set_page_config(page_title="EmotionVision Combined Test", page_icon="😊")

st.title("EmotionVision - Combined Webcam Test")

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
st.success("Emotion model loaded.")

def predict_emotion(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    predictions = model.predict(arr, verbose=0)[0]

    index = int(np.argmax(predictions))
    probabilities = {
        label: float(predictions[i]) * 100
        for i, label in enumerate(CLASS_NAMES)
    }
    return CLASS_NAMES[index], float(predictions[index]) * 100, probabilities

class EmotionTestProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            face, bbox = detect_face(pil_image)

            if face is None or bbox is None:
                cv2.putText(
                    img, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            else:
                x, y, width, height = map(int, bbox)

                cv2.rectangle(
                    img, (x, y), (x + width, y + height),
                    (0, 255, 0), 2
                )

                emotion, confidence, probabilities = predict_emotion(face)

                cv2.putText(
                    img, f"{emotion}: {confidence:.1f}%",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

                landmarks = detect_landmarks(pil_image)
                if landmarks:
                    for point in landmarks:
                        px = int(point["x"] * img.shape[1])
                        py = int(point["y"] * img.shape[0])
                        cv2.circle(img, (px, py), 1, (0, 255, 255), -1)

        except Exception as e:
            print(f"Webcam frame processing error: {type(e).__name__}: {e}")
            cv2.putText(
                img, "Processing error - check app logs", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="emotionvision-combined-test",
    video_processor_factory=EmotionTestProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
