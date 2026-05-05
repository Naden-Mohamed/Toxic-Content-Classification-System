from config import Config
from imagecaption import ImageCaptioning
from toxicity_classifier import ToxicityPipeline
import streamlit as st
from PIL import Image

@st.cache_resource
def get_models():
    """Load models once and keep them in memory."""
    config = Config()
    cap_model = ImageCaptioning(model_name=config.PRETRAINED_CAPTIONING_MODEL)
    tox_pipeline = ToxicityPipeline.load("saved_model.pt", "vocab.pkl", config)
    return cap_model, tox_pipeline

def main():
    with st.spinner("Loading AI Models... Please wait."):
        image_captioning, toxicity_pipeline = get_models()

    st.set_page_config(page_title="Neurova Toxic Content Classifier", layout="wide")


    st.title("Content Safety Classifier")
    st.write("Analyze text queries or images using BiLSTM.")

    tab_text, tab_image = st.tabs(["💬 Text Query", "🖼️ Image Upload"])

    result = None

    with tab_text:
        user_text = st.text_area("Enter text to analyze:", key="text_input")
        if st.button("Analyze Text", type="primary"):
            if user_text:
                result = toxicity_pipeline.predict(user_text)
            else:
                st.warning("Please enter some text first.")

    with tab_image:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("Process Image", type="primary"):
                with st.spinner("Generating caption..."):
                    caption = image_captioning.generate_caption(image) 
                    st.info(f"**Generated Caption:** {caption}")
                    result = toxicity_pipeline.predict(caption)

    if result:
        st.divider()
        st.subheader("Analysis Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Category", result['label'])
        with col2:
            st.metric("Confidence Score", f"{result['confidence']*100:.2f}%")
        with col3:
            status = "✅ PASS" if result['label'] == "Safe" else "⚠️ FLAG"
            st.metric("Safety Status", status)

if __name__ == "__main__":
    main()