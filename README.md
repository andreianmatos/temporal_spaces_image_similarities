# Temporal Spaces Image Similarities

This repository demonstrates image similarity using a pre-trained ResNet50 model. It compares a current person's image with images from a dataset of people, finding the most similar image.

## Project Structure

- **image_similarity_package/**
  - `__init__.py` : file that makes it a package.
  - `image_similarity.py`: python code with functions for image similarity operations.
- **notebooks/**
  - `similarities.ipynb`: Jupyter Notebook for comparing the current person's movement texture to the past ones.
  - `package_usage_example.ipynb`: Jupyter Notebook that exemplifies the functionality by calling the package.
- **data/**
  - `people.zip`: Texture of past people, a folder per person, containing its movement texture image.
  - `people_imgs.zip`: Texture of past people, all images in the same folder.
  - `current_person.png`: Texture image of current person's movement.
