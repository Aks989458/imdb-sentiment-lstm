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
st.write("LSTM-based sentiment analysis model")

# -----------------------------
# Load Model + Vocab (SAFE)
# -----------------------------
@st.cache_resource
def load_model_and_vocab():
    # Load vocabulary
    if not os.path.exists("vocab.pkl"):
        st.error("❌ vocab.pkl not found")
        st.stop()

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # Load model weights
    if not os.path.exists("model_weights.pth"):
        st.error("❌ model_weights.pth not found")
        st.stop()

    model = SentimentLSTM(vocab_size=len(vocab))

    state_dict = torch.load(
        "model_weights.pth",
        map_location="cpu"
    )
    model.load_state_dict(state_dict)

    model.eval()
    device = torch.device("cpu")  # Streamlit Cloud = CPU only
    model.to(device)

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

    x = torch.tensor(
        encoded,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

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
        st.info(f"Confidence Score: **{prob:.2f}")

st.markdown("---")
st.caption("Model: LSTM | Dataset: IMDB | Framework: PyTorch")
