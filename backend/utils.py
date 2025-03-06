import os
import yaml
import cv2
import numpy as np

def ensure_directories(path):
    os.makedirs(f"{path}/images", exist_ok=True)
    os.makedirs(f"{path}/labels", exist_ok=True)

def save_image(image_bytes, path):
    with open(path, "wb") as img_file:
        img_file.write(image_bytes)

def save_labels(labels, path):
    with open(path, "w") as label_file:
        label_file.write(labels)

def update_data_yaml(path, classes):
    data_yaml = {"path": path, "train": "images", "val": "images", "nc": len(classes), "names": classes}
    with open(f"{path}/data.yaml", "w") as file:
        yaml.dump(data_yaml, file)