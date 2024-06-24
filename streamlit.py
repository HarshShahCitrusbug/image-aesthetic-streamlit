from PIL import Image
import torch
import streamlit as st
from transformers import AutoProcessor, CLIPModel
import requests
import pickle
import numpy as np
from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

cities_list = ["Ahmedabad"]


prompt = f"""Given a list of cities, Provide me a single list only containing both the famous tourist spots and their corresponding categories for those cities. Ensure the result is in the array data structure and no other extra characters or context.\n\nList of Cities: {cities_list}\n\n## Output/Response Example :\n[\n[<city_name1>, <location_name of city_name1>, <category_name of location>],\n[<city_name2>, <location_name of city_name2>, <category_name of location>],\n]"""

openai_client = OpenAI(api_key=st.secrets["openai_api_key"])
openai_response = openai_client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": prompt}],
)
response = openai_response.choices[0].message.content
response = eval(response)

positive_classes = []
negative_classes = []

for item in response:
    if len(item) == 2:
        location = item[0]
        location_category = item[1]

        positive_classes.append(
            f"an outstanding picture of the {location_category} at #{location}"
        )
        negative_classes.append(
            f"a horrible picture of the {location_category} at #{location}"
        )
    elif len(item) == 3:
        city_name = item[0]
        location = item[1]
        location_category = item[2]

        positive_classes.append(
            f"an outstanding picture of the {location_category} at #{location}, {city_name}"
        )
        negative_classes.append(
            f"a horrible picture of the {location_category} at #{location}, {city_name}"
        )

if len(positive_classes) and len(negative_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    def load_image_PIL(url_or_path):
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            return Image.open(requests.get(url_or_path, stream=True).raw)
        else:
            return Image.open(url_or_path)

    def cosine_similarity(vec1, vec2):
        # Compute the dot product of vec1 and vec2
        dot_product = np.dot(vec1, vec2)

        # Compute the L2 norm of vec1 and vec2
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        # Compute the cosine similarity
        similarity = dot_product / (norm_vec1 * norm_vec2)

        return similarity

    positive_inputs = processor(
        text=positive_classes, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        positive_text_features = model.get_text_features(**positive_inputs)
    negative_inputs = processor(
        text=negative_classes, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        negative_text_features = model.get_text_features(**negative_inputs)

    positive_prompt_vectors = np.array(positive_text_features)
    # Compute the average vector
    average_positive_vector = np.mean(positive_prompt_vectors, axis=0)

    negative_prompt_vectors = np.array(negative_text_features)
    # Compute the average vector
    average_negative_vector = np.mean(negative_prompt_vectors, axis=0)

    with open("positive_prompt.pkl", "wb") as f:
        pickle.dump(average_positive_vector, f)
    with open("negative_prompt.pkl", "wb") as f:
        pickle.dump(average_negative_vector, f)

    with open("positive_prompt.pkl", "rb") as f:
        average_positive_vector = pickle.load(f)
    with open("negative_prompt.pkl", "rb") as f:
        average_negative_vector = pickle.load(f)

    def predict(img_url):
        image1 = img_url
        with torch.no_grad():
            inputs1 = processor(images=image1, return_tensors="pt").to(device)
            image_features1 = model.get_image_features(**inputs1)
        image_vector = image_features1.numpy()
        positive_similarity = cosine_similarity(
            average_positive_vector, np.transpose(image_vector)
        )
        negative_similarity = cosine_similarity(
            average_negative_vector, np.transpose(image_vector)
        )
        aesthetic_score = (+1 * positive_similarity) + (-1 * negative_similarity)
        return aesthetic_score * 1000


if "user_input" not in st.session_state:
    st.session_state.user_input = ""


def submit():
    st.session_state.user_input = st.session_state.cities_list
    st.session_state.cities_list = ""


st.header("Image Aesthetics Scorer")

st.text_input("Enter cities list: ", key="cities_list", on_change=submit)
user_input = st.session_state.user_input
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
picture_width = st.sidebar.slider("Picture Width", min_value=100, max_value=500)
if uploaded_file is not None and user_input:
    image = Image.open(uploaded_file)
    st.subheader("Input", divider="rainbow")
    st.image(image, caption="Uploaded Image", width=picture_width)

    # Call your function with the uploaded image
    results = predict(image)

    st.subheader("Results", divider="rainbow")
    # Display the results
    st.image(image, caption=results, width=picture_width)
