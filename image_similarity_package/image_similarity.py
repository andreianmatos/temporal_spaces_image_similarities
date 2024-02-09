from zipfile import ZipFile
import requests
from io import BytesIO
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def download_and_extract_zip(zip_link, destination_folder):
    response = requests.get(zip_link)
    zip_data = BytesIO(response.content)
    with ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

def get_image_embedding(img_path, base_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embeddings = base_model.predict(img_array)
    return embeddings.flatten()

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def find_most_similar_image(query_image_path, dataset_folder, base_model):
    query_image = Image.open(query_image_path)
    query_embedding = get_image_embedding(query_image_path, base_model)

    most_similar_image_path = None
    max_similarity = -1

    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.lower().endswith(('.png')):
                img_path = os.path.join(root, file)
                img_embedding = get_image_embedding(img_path, base_model)
                similarity_score = cosine_similarity(query_embedding, img_embedding)

                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    most_similar_image_path = img_path

    return most_similar_image_path, query_image, max_similarity

def plot_images(query_image, similar_image, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(query_image)
    axs[0].set_title(title1)
    axs[1].imshow(similar_image)
    axs[1].set_title(title2)
    plt.show()
