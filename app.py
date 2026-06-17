import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── PAGE CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Fresh vs Stale Vegetable Classifier", page_icon="🥦")

st.title("🥦 Fresh vs Stale Vegetable Classifier")
st.write("Upload an image of a vegetable and the model will predict whether it's **fresh** or **stale**.")

# ── SETTINGS ────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
MODEL_PATH = "fresh_stale_model.pth"
CLASSES = ["fresh", "stale"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LOAD MODEL (cached so it only loads once) ──────────────────────────
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


model = load_model()

# ── IMAGE TRANSFORM ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# ── FILE UPLOAD ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        label = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # Display result
    st.markdown("---")
    if label == "fresh":
        st.success(f"### ✅ Prediction: FRESH ({confidence:.1f}% confident)")
    else:
        st.error(f"### ⚠️ Prediction: STALE ({confidence:.1f}% confident)")

    # Show probability breakdown
    st.write("**Confidence breakdown:**")
    st.progress(int(probs[0].item() * 100))
    st.write(f"Fresh: {probs[0].item()*100:.1f}%")
    st.progress(int(probs[1].item() * 100))
    st.write(f"Stale: {probs[1].item()*100:.1f}%")

st.markdown("---")
st.caption("Model: MobileNetV2 (transfer learning) trained on the Fresh and Stale Vegetable dataset.")
