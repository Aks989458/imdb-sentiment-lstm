import os
import pickle
import torch
import streamlit as st
from model import SentimentLSTM

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered"
)

st.title("🎬 IMDB Sentiment Analysis Dashboard")
st.write("LSTM model trained using Lightning")

# -----------------------------
# Load Model + Vocab (Cached)
# -----------------------------
@st.cache_resource
def load_model_and_vocab():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    ckpt_dir = "Old-checkpoints"

    if not os.path.exists(ckpt_dir):
        st.error(f"❌ Checkpoint directory '{ckpt_dir}' not found.")
        st.stop()

    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    if not ckpt_files:
        st.error("❌ No checkpoint file found.")
        st.stop()

    ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])

    model = SentimentLSTM.load_from_checkpoint(
        ckpt_path,
        vocab_size=len(vocab),
        weights_only=False  # 🔥 CRITICAL FIX
    )

    device = torch.device("cpu")  # Streamlit Cloud = CPU only
    model.to(device)
    model.eval()

    return model, vocab, device


model, vocab, device = load_model_and_vocab()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):
    if text.strip() == "":
        return None, None

    tokens = text.lower().split()
    encoded = [vocab.get(w, vocab["<unk>"]) for w in tokens]

    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "Positive 😀" if prob > 0.5 else "Negative 😞"
    return prob, label

# -----------------------------
# UI
# -----------------------------
user_text = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="The movie was absolutely amazing with great acting..."
)

if st.button("Predict Sentiment"):
    prob, label = predict_sentiment(user_text)

    if prob is None:
        st.warning("Please enter some text.")
    else:
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence Score: **{prob:.2f}**")

st.markdown("---")
st.caption("Model: LSTM | Dataset: IMDB | Framework: Lightning")
