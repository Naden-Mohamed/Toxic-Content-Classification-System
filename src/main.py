from configs.config import Config
from imagecaption.imagecaption import ImageCaptioning
from toxicity_classifier import ToxicityPipeline
import streamlit as st
from PIL import Image
from database import Database

@st.cache_resource
def get_models():
    """Load models once and keep them in memory."""
    config = Config()
<<<<<<< HEAD
    cap_model = ImageCaptioning(model_name=config.PRETRAINED_CAPTIONING_MODEL)
    tox_pipeline = ToxicityPipeline.load("saved_model.pt", "vocab.pkl", config)
    db = Database() 
    return cap_model, tox_pipeline, db
=======
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    print(f"Device: {config.DEVICE}")

    print("\nLoading & preprocessing data …")
    df = pd.read_excel("/dataset/NLP_Neurova_toxic_content_classification.xlsx")
    print(f"  Dataset size : {len(df):,}")
    print(f"  Label distribution:\n{df['Toxic Category'].value_counts().to_string()}")
    le = LabelEncoder()
    df["Toxic Category"] = le.fit_transform(df["Toxic Category"])
    preprocessor = TextPreprocessor()
    df["clean_text"] = preprocessor.clean_batch(df["query"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["Toxic Category"].tolist(),
        test_size   = config.TEST_SIZE,
        random_state= config.RANDOM_SEED,
        stratify    = df["Toxic Category"],
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    print("\n Building vocabulary …")
    vocab = Vocabulary(max_size=config.MAX_VOCAB_SIZE, min_freq=config.MIN_WORD_FREQ)
    vocab.build([vocab.tokenization(t) for t in X_train])
    vocab.save(config.VOCAB_PATH)

    print("\n Creating datasets …")
    train_ds = ToxicDataset(X_train, y_train, vocab, config.MAX_SEQ_LEN, preprocessor)
    test_ds  = ToxicDataset(X_test,  y_test,  vocab, config.MAX_SEQ_LEN, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)

    NUM_CLASSES = df["Toxic Category"].nunique()

    print("\n Building model …")
    model = BRNN(
        vocab_size    = len(vocab),
        embedding_dim = config.EMBEDDING_DIM,
        hidden_dim    = config.HIDDEN_DIM,
        output_dim    = NUM_CLASSES,
        num_layers    = config.NUM_LAYERS,
        dropout       = config.DROPOUT,
        bidirectional = config.BIDIRECTIONAL,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(model)

    print("\nTraining …")
    trainer = Trainer(model, config)
    trainer.train(train_loader, test_loader)
    ToxicityPipeline.plot_training_history(trainer.history, config.RESULTS_DIR)

    print("\n Evaluating …")
    evaluator = Evaluator(model, config)
    results   = evaluator.evaluate(test_loader)

    results_path = f"{config.RESULTS_DIR}/metrics.json"
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "confusion_matrix"}, f, indent=2)
    print(f"[Eval] Metrics saved to {results_path}")
>>>>>>> 37bfa47674e4f2afec2abbdceba9893a811dadcb


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
