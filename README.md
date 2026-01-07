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

imdb-sentiment-lstm/
│
├── app.py # Streamlit dashboard
├── model.py # LSTM model definition
├── vocab.pkl # Saved vocabulary
├── model_weights.pth # Trained model weights (state_dict)
├── requirements.txt # Dependencies
└── README.md

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Lightning (for training)  
- Streamlit (for deployment)  
- Scikit-learn  
- Git & GitHub  

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Aks989458/imdb-sentiment-lstm.git
cd imdb-sentiment-lstm
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---
### 📌 Deployment Notes

The deployed version uses model_weights.pth instead of Lightning .ckpt files

This avoids Python and pickle compatibility issues in production

CPU-only inference is used for Streamlit Cloud compatibility

---

### 🎓 Key Learning Outcomes

Building NLP pipelines using deep learning

Handling real-world model serialization and deployment issues

Deploying ML models as public web applications

Managing large model files and production constraints

---

### 🧾 License

This project is intended for educational and academic use.

---
