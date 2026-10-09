
import streamlit as st
import os

st.set_page_config(
    page_title="EmotionVision Diagnostic",
    page_icon="😊"
)

st.title("EmotionVision - Startup Test")
st.write("Step 1: Streamlit started successfully.")

try:
    st.write("Step 2: Importing TensorFlow...")
    import tensorflow as tf
    st.success(f"TensorFlow imported: {tf.__version__}")

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ai_model",
        "emotion_model.keras"
    )

    st.write(f"Step 3: Checking model file: `{model_path}`")

    if not os.path.exists(model_path):
        st.error("Model file was not found at this path.")
    else:
        st.success("Model file exists.")

        st.write("Step 4: Loading emotion model...")
        model = tf.keras.models.load_model(model_path)

        st.success("Emotion model loaded successfully!")
        st.write("Model input shape:", model.input_shape)
        st.write("Model output shape:", model.output_shape)

except Exception as e:
    st.exception(e)
