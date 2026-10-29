# Plant Leaf Disease Classification (CNN)

Classifying plant leaf diseases from images with a **TensorFlow/Keras CNN** trained on directory-structured image data.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)

## Overview

## Origin

This project was originally developed inside the `Data-Science` notebook collection and now lives in its own repository for easier discovery and use.

The notebook builds a convolutional neural network (`Sequential` model with `Conv2D` layers) and trains it with Keras `ImageDataGenerator` / `flow_from_directory`, the standard PlantVillage-style setup for leaf-disease recognition — a practical computer-vision project for agriculture.

## What's Inside

- `leaf_disease_prediction.ipynb` — image loading & augmentation → CNN build → training → evaluation

## Getting Started

```bash
git clone https://github.com/datascientist970/plant-leaf-disease-classification.git
cd plant-leaf-disease-classification
pip install -r requirements.txt
jupyter notebook leaf_disease_prediction.ipynb
```

You need an image dataset arranged in class folders (one folder per disease class) — the dataset is not included in this repository; update the data path in the first cells to your copy.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · NumPy · Matplotlib

## Author

**Muhammad Dawood** — AI & Data Science Engineer
GitHub: https://github.com/datascientist970 · LinkedIn: https://www.linkedin.com/in/dawood406/
Mardan, Khyber Pakhtunkhwa, Pakistan
