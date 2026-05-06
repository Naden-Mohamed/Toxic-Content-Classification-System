from config import Config
from imagecaption import ImageCaptioning
from toxicity_classifier import ToxicityPipeline
import streamlit as st
from PIL import Image
from database import Database

@st.cache_resource
def get_models():
    """Load models once and keep them in memory."""
    config = Config()
    cap_model = ImageCaptioning(model_name=config.PRETRAINED_CAPTIONING_MODEL)
    tox_pipeline = ToxicityPipeline.load("saved_model.pt", "vocab.pkl", config)
    db = Database() 
    return cap_model, tox_pipeline, db


def main():
    st.set_page_config(page_title="Neurova Toxic Content Classifier", layout="wide")

    with st.spinner("Loading AI Models... Please wait."):
        image_captioning, toxicity_pipeline, db = get_models()

    st.title("Content Safety Classifier")
    st.write("Analyze text queries or images using BiLSTM.")

    tab_text, tab_image = st.tabs(["💬 Text Query", "🖼️ Image Upload"])

    result = None

    with tab_text:
        with st.form("text_form"):  
            user_text = st.text_area("Enter text to analyze:")
            submitted = st.form_submit_button("Analyze Text")

        if submitted:
            if user_text.strip():
                try:
                    result = toxicity_pipeline.predict(user_text)
                    db.append_to_csv(user_text, result) 
                    db.insert_record(user_text, result)  
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter some text first.")


    with tab_image:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB") 
            st.image(image, caption="Uploaded Image", width=400)

            if st.button("Process Image"):
                with st.spinner("Generating caption..."):
                    try:
                        caption = image_captioning.generate_caption(image)
                        st.info(f"**Generated Caption:** {caption}")

                        result = toxicity_pipeline.predict(caption)

                        db.append_to_csv(caption, result)   
                        db.insert_record(caption, result)  

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    if result:
        st.divider()
        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Detected Category", result['label'])
        col2.metric("Confidence Score", f"{result['confidence']*100:.2f}%")

        status = "✅ PASS" if result['label'] == "Safe" else "⚠️ FLAG"
        col3.metric("Safety Status", status)


if __name__ == "__main__":
    main()