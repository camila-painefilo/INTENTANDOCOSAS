import streamlit as st
import torch
import clip
from PIL import Image
import os

# 1. Load CLIP model and device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. List images in your folder
IMAGE_FOLDER = "Images"
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

@st.cache_resource
def compute_image_features(image_files):
    features = []
    for fname in image_files:
        image = preprocess(Image.open(os.path.join(IMAGE_FOLDER, fname))).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image)
        features.append(feature.cpu())
    return torch.cat(features, dim=0)

image_features = compute_image_features(image_files)

# 3. Streamlit UI
st.title("CLIP Zero-Shot Image Search")
query = st.text_input("Type your search (e.g., 'group of people at the beach'):")

if query:
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(device)
        text_features = model.encode_text(text_tokens).cpu()

    # Compute cosine similarity
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    similarities = (image_features_norm @ text_features_norm.T).squeeze(1)

    # Show top 5 results
    top_indices = similarities.argsort(descending=True)[:5]
    st.subheader("Top matches:")
    for idx in top_indices:
        st.image(os.path.join(IMAGE_FOLDER, image_files[idx]), caption=image_files[idx], width=300)
