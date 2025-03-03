import os
import rawpy
import imageio
import numpy as np
import tensorflow as tf

def load_dataset(raw_folder, jpeg_folder, image_size=(512, 512)):
    raw_paths = sorted([os.path.join(raw_folder, f) for f in os.listdir(raw_folder) if f.lower().endswith(('.arw', '.cr2', '.dng', '.nef', '.raw'))])
    jpeg_paths = sorted([os.path.join(jpeg_folder, f) for f in os.listdir(jpeg_folder) if f.lower().endswith('.jpg')])

    X, Y = [], []

    for raw_path, jpeg_path in zip(raw_paths, jpeg_paths):
        try:
            with rawpy.imread(raw_path) as raw:
                raw_image = raw.postprocess()
                raw_image = tf.image.resize(raw_image, image_size) / 255.0 
                X.append(tf.keras.utils.img_to_array(raw_image))

            jpeg_image = imageio.imread(jpeg_path)
            jpeg_image = tf.image.resize(jpeg_image, image_size) / 255.0  
            Y.append(tf.keras.utils.img_to_array(jpeg_image))

        except Exception as e:
            print(f"Erreur lors du chargement de {raw_path} ou {jpeg_path} : {e}")

    return np.array(X), np.array(Y)

raw_folder = "dataset/raw"
jpeg_folder = "dataset/traited"

X_train, Y_train = load_dataset(raw_folder, jpeg_folder)
print(f"Dataset chargé : {X_train.shape} RAW → {Y_train.shape} JPEG")

