# 🥦 Fresh vs Stale Vegetable Classifier

A web app that classifies vegetable images as **fresh** or **stale** using a MobileNetV2 model
(transfer learning), built with PyTorch and deployed with Streamlit.

---

## 🚀 Live Demo

Once deployed (see steps below), your live link will look like:
`https://your-app-name.streamlit.app`

---

## 📁 Project Files

```
fresh-stale-app/
├── app.py                   # Streamlit web app
├── requirements.txt         # Python dependencies
├── fresh_stale_model.pth    # Your trained model (ADD THIS YOURSELF)
└── README.md                # This file
```

---

## 🛠️ Step 1: Upload to GitHub

1. Go to [github.com](https://github.com) and log in (or create a free account)
2. Click **New repository** (top right → "+" → "New repository")
3. Name it something like `fresh-stale-classifier`
4. Set it to **Public** (required for free Streamlit hosting)
5. Click **Create repository**
6. Click **uploading an existing file**
7. Drag and drop these files into the repo:
   - `app.py`
   - `requirements.txt`
   - `fresh_stale_model.pth` (your trained model file from your Desktop)
8. Click **Commit changes**

> ⚠️ Note: `fresh_stale_model.pth` may be a large file (10-15 MB typically, which GitHub
> allows fine via the web upload — anything under 25 MB works through the browser).

---

## 🌐 Step 2: Deploy on Streamlit Community Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Choose your repository: `fresh-stale-classifier`
5. Set the main file path to: `app.py`
6. Click **Deploy**

Wait 1-2 minutes while it installs dependencies and starts the app. You'll get a live
public link you can open on any device and share in your presentation.

---

## 🖥️ Step 3 (Optional): Run It Locally First to Test

Before deploying, you can test it on your own computer:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

- **Model**: MobileNetV2, pretrained on ImageNet, fine-tuned on the Fresh and Stale
  Vegetable dataset (transfer learning — only the final layer was retrained)
- **Input**: Any vegetable image (jpg/jpeg/png)
- **Output**: Predicted label (`fresh` or `stale`) with a confidence percentage

---

## 📊 Dataset

Trained on the [Fresh and Stale Classification dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
from Kaggle, containing fresh and stale images of apples, bananas, bitter gourd, capsicum,
oranges, and tomatoes.
