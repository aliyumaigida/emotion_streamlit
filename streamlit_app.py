
import streamlit as st

st.set_page_config(
    page_title="EmotionVision - Webcam Test",
    page_icon="😊"
)

st.title("EmotionVision - Webcam Test")

try:
    st.write("Step 1: Importing AV...")
    import av
    st.success(f"AV imported: {av.__version__}")

    st.write("Step 2: Importing Streamlit WebRTC...")
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase
    )
    st.success("Streamlit WebRTC imported successfully.")

    class TestVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            return frame

    st.write("Step 3: Starting webcam component...")

    webrtc_streamer(
        key="webcam-diagnostic",
        video_processor_factory=TestVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.success("Webcam component initialized.")

except Exception as e:
    st.exception(e)
