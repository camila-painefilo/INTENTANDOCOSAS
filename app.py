import streamlit as st
import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Candidate topics for image-to-text matching
topics = [
    "a dog",
    "a group of people at the beach",
    "a pizza",
    "a sunset",
    "a person riding a bike",
    "a city street",
    "a laptop",
    "a cat",
    "a mountain landscape"
]

# --- INTERFACE ---

st.title("CLIP Image & Text Search Demo")

option = st.radio(
    "What do you want to do?",
    ("Upload an image and get its topic", "Search images by text")
)

# --- 1. Upload image and get the topic ---
if option == "Upload an image and get its topic":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", width=300)

        # Preprocess and encode image
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).cpu()

        # Encode candidate topics
        with torch.no_grad():
            text_tokens = clip.tokenize(topics).to(device)
            text_features = model.encode_text(text_tokens).cpu()

        # Compute similarities
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)
        best_idx = similarities.argmax().item()
        st.success(f"**Best match:** {topics[best_idx]}")
        # Optionally, show all scores
        st.write("All topics and scores:")
        results = sorted(zip(topics, similarities.tolist()), key=lambda x: x[1], reverse=True)
        for topic, score in results:
            st.write(f"{topic}: {score:.3f}")

# --- 2. Text search as before ---
elif option == "Search images by text":
    IMAGE_FOLDER = "images"  # Or "Images", depending on your folder name!
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
