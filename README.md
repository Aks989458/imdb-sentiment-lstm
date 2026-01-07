# 🎬 IMDB Sentiment Analysis using LSTM

This project implements an end-to-end **sentiment analysis system** for IMDB movie reviews using a **Long Short-Term Memory (LSTM)** neural network built with PyTorch. The trained model is deployed as a **public Streamlit web dashboard** for real-time sentiment prediction.

---

## 🚀 Project Overview

- Built an LSTM-based deep learning model to classify movie reviews as **Positive** or **Negative**
- Trained and evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix
- Converted the trained Lightning checkpoint into a lightweight `state_dict` for production deployment
- Deployed the model as a **permanent public web app** using Streamlit Cloud

---

## 🧠 Model Architecture

- **Embedding Layer** – Converts word indices into dense vectors  
- **LSTM Layer** – Captures sequential and contextual information  
- **Fully Connected Layer** – Outputs sentiment logits  
- **Sigmoid Activation** – Produces probability score  

Loss Function: `BCEWithLogitsLoss`  
Optimizer: `Adam`

---

## 📊 Evaluation Metrics

The model was evaluated on an unseen test set using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (optional)

These metrics ensure robust and unbiased performance evaluation.

---

## 🌐 Live Demo

The application is deployed permanently using **Streamlit Community Cloud**.

🔗 **Live App URL:**  
*(https://imdb-sentiment-lstm-fzyy5sqvdupmtehq85xict.streamlit.app/)*

Users can enter a movie review and instantly receive:
- Sentiment prediction (Positive / Negative)
- Confidence score

---

## 📁 Repository Structure

