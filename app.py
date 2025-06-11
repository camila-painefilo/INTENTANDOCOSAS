import streamlit as st
import torch
import clip
from PIL import Image
import os

# Set IBALab-inspired page config
st.set_page_config(
    page_title="IBALab-Inspired CLIP Image Search",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- IBALab-inspired CSS for look and feel ---
st.markdown("""
    <style>
    body {
        background-color: #eaf5fd;
    }
    .main {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 32px 24px 24px 24px;
        box-shadow: 0 4px 24px 0 rgba(33, 82, 161, 0.06);
    }
    h1 {
        background: linear-gradient(90deg, #205fbc 0%, #429fe2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6em !important;
        margin-bottom: 10px;
    }
    .stRadio > div {
        background-color: #f7faff;
        border-radius: 10px;
        padding: 10px 18px;
    }
    .stButton button {
        color: white;
        background: linear-gradient(90deg, #205fbc 0%, #429fe2 100%);
        border: none;
        border-radius: 7px;
        font-weight: 600;
        font-size: 1.1em;
        padding: 0.5em 1.2em;
        transition: box-shadow .2s;
        box-shadow: 0 2px 8px 0 #2152a11a;
    }
    .result-card {
        background-color: #f7faff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px 0 #2152a11a;
        margin-bottom: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CLIP setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Candidate topics for image-to-text matching ---
topics = [
    "a dog", "a cat", "a horse", "a bird", "a group of people at the beach", "a sunset over the ocean",
    "a city skyline at night", "a busy street market", "a mountain landscape", "a snowy forest",
    "a child playing with a toy", "a family having a picnic", "a couple taking a selfie",
    "a car on a highway", "a train at the station", "an airplane flying in the sky",
    "a bowl of fruit", "a pizza on a table", "a birthday cake", "a cup of coffee",
    "people riding bicycles", "a man wearing a suit", "a woman in a red dress", "a baby sleeping",
    "a person swimming in a pool", "a person hiking up a hill", "a soccer game", "a basketball player",
    "a baseball stadium", "a tennis match", "a group of friends camping", "a campfire at night",
    "a computer on a desk", "a smartphone on a table", "a person using a laptop", "a person reading a book",
    "a stack of books", "a painting in a museum", "a sculpture in a park", "a person taking a photograph",
    "a street filled with cars", "a bus stop", "a subway station", "a farmer working in a field",
    "a cow grazing", "a flock of sheep", "a chicken coop", "a person fishing in a river",
    "a sailboat on a lake", "a bridge over a river", "a lighthouse by the sea", "a castle on a hill",
    "a waterfall in a forest", "a desert with sand dunes", "a cactus in the desert", "a palm tree on a beach",
    "a glacier in the mountains", "a volcano erupting", "a rainbow in the sky", "a thunderstorm",
    "a close-up of a flower", "a bee on a flower", "a butterfly on a leaf", "a frog on a lily pad",
    "a monkey in a tree", "an elephant in the savanna", "a lion resting", "a tiger walking", "a bear fishing",
    "a panda eating bamboo", "a giraffe eating leaves", "a zebra in the grass", "a kangaroo in Australia",
    "a penguin on ice", "a polar bear", "a whale jumping out of the water", "a dolphin swimming",
    "a shark underwater", "a turtle on the beach", "a crab in the sand", "a jellyfish in the ocean",
    "a person riding a horse", "a person skiing down a slope", "a person snowboarding",
    "a person surfing a wave", "a person skateboarding", "a person rock climbing", "a person skydiving",
    "a person doing yoga", "a person meditating", "a person playing guitar", "a person playing piano",
    "a band performing on stage", "a singer with a microphone", "a dancer in a studio", "a ballet performance",
    "a theater stage", "an audience clapping", "a parade on the street", "a festival with fireworks",
    "a food truck", "a street vendor selling food", "a grocery store", "a bakery with bread",
    "a chef cooking in a kitchen", "a waiter serving food", "a restaurant menu", "a salad in a bowl",
    "a sandwich on a plate", "a hamburger and fries", "a hot dog at a stadium", "a cup of tea",
    "a glass of orange juice", "a bottle of wine", "a cocktail at a bar", "a wedding ceremony",
    "a bride and groom", "a birthday party", "a graduation ceremony", "a school classroom",
    "a chalkboard with writing", "a teacher giving a lesson", "students raising their hands",
    "a playground with children", "a soccer field", "a basketball court", "a swimming pool",
    "a gym with exercise equipment", "a running track", "a yoga mat", "a park with trees",
    "a bench in a park", "a bicycle lane", "a train crossing", "a taxi on the street",
    "an ambulance", "a police car", "a fire truck", "a hospital building", "a pharmacy",
    "a dentist office", "a doctor examining a patient", "a nurse helping a patient",
    "a baby in a stroller", "a toddler learning to walk", "an elderly person with a cane",
    "a person in a wheelchair", "a blind person with a guide dog", "a person painting a wall",
    "a person planting a tree", "a person mowing the lawn", "a gardener watering plants",
    "a person shoveling snow", "a person walking a dog", "a person jogging in the park",
    "a couple holding hands", "a group of teenagers", "a family reunion", "a person wearing sunglasses",
    "a person with a hat", "a person with an umbrella", "a person wearing a backpack", "a suitcase at the airport",
    "an airplane at the gate", "luggage on a conveyor belt", "a passport and tickets", "a world map",
    "a globe on a desk", "a calendar on the wall", "a clock showing time", "a watch on a wrist",
    "a lamp on a nightstand", "a bed with pillows", "a sofa in a living room", "a television on the wall",
    "a kitchen with appliances", "a dining table with chairs", "a bathtub in a bathroom", "a mirror on the wall",
    "a closet with clothes", "a shoe rack", "a laundry basket", "a washing machine", "a vacuum cleaner",
    "a car in a garage", "a bicycle in a shed", "a lawnmower in a yard", "a fence around a house",
    "a mailbox by the road", "a street sign", "a traffic light", "a crosswalk", "a highway overpass",
    "a construction site", "a crane lifting materials", "a building under construction", "a skyscraper",
    "an office building", "a bank on the corner", "a post office", "a courthouse", "a library with books",
    "a museum with exhibits", "an art gallery", "a science laboratory", "a classroom with students",
    "a school bus", "a playground slide", "a sandbox", "a swing set", "a merry-go-round",
    "a roller coaster", "a ferris wheel", "a carnival ride", "a ticket booth", "a crowd at a concert",
    "a movie theater", "a popcorn stand", "a person watching a movie", "a person playing video games",
    "a computer screen", "a tablet device", "a smartphone with apps", "a printer on a desk", "a mouse and keyboard",
    "a server room", "a security camera", "a drone flying", "a robot vacuum", "a smart speaker"
]

# --- Centered main content ---
with st.container():
    st.markdown('<h1>CLIP Image & Topic Explorer</h1>', unsafe_allow_html=True)
    st.write(
        "<span style='color:#555555; font-size:1.1em;'>Experience IBALab-inspired image search using AI. Search by text, or upload an image to see what it's about!</span>",
        unsafe_allow_html=True,
    )

    option = st.radio(
        "Choose an action:",
        ("🔎 Search images by text", "🖼️ Upload an image to get its topic"),
        horizontal=True,
    )

    st.write("")

    # --- 1. Text search as before ---
    if option == "🔎 Search images by text":
        IMAGE_FOLDER = "Images"  # Or "images", depending on your folder name!
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

        query = st.text_input("Type your search (e.g., 'group of people at the beach'):", key="query")
        if query:
            with torch.no_grad():
                text_tokens = clip.tokenize([query]).to(device)
                text_features = model.encode_text(text_tokens).cpu()

            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features_norm @ text_features_norm.T).squeeze(1)
            top_indices = similarities.argsort(descending=True)[:5]
            st.markdown('<div class="result-card"><b>Top matches:</b></div>', unsafe_allow_html=True)
            for idx in top_indices:
                st.image(os.path.join(IMAGE_FOLDER, image_files[idx]), caption=image_files[idx], width=300)

    # --- 2. Upload image and get the topic ---
    elif option == "🖼️ Upload an image to get its topic":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="fileuploader")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded image", width=320)

            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image_input).cpu()
                text_tokens = clip.tokenize(topics).to(device)
                text_features = model.encode_text(text_tokens).cpu()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(0)
            best_idx = similarities.argmax().item()

            st.markdown(
                f'<div class="result-card"><b>Best match:</b> <span style="color:#205fbc;">{topics[best_idx]}</span></div>',
                unsafe_allow_html=True
            )
            with st.expander("See all topics and scores"):
                results = sorted(zip(topics, similarities.tolist()), key=lambda x: x[1], reverse=True)
                for topic, score in results:
                    st.write(f"{topic}: {score:.3f}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("<center><small>Inspired by <a href='https://ibalab.quv.kr/' target='_blank'>IBALab</a> • Powered by OpenAI CLIP & Streamlit</small></center>", unsafe_allow_html=True)

