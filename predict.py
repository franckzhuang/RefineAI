import tensorflow as tf
import rawpy
import imageio.v3 as iio
import numpy as np
import cv2
import os
from utils import load_image, preprocess_image, logger, calculate_psnr, calculate_ssim 

IMAGE_SIZE = (256, 256) 
MODEL_PATH = "saved_model/my_model"
INPUT_RAW_DIR = "data/raw_to_predict"
OUTPUT_DIR = "data/predicted_images"

def predict_raw_images(model_path, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = tf.keras.models.load_model(model_path, custom_objects={'calculate_psnr': calculate_psnr, 'calculate_ssim': calculate_ssim})
    logger.info(f"Modèle chargé depuis {model_path}")

    num_images = len([f for f in os.listdir(input_dir) if f.lower().endswith('.cr2')])
    processed_count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.cr2'):
            filepath = os.path.join(input_dir, filename)
            logger.info(f"Traitement de {filename} ({processed_count + 1}/{num_images})...")
            try:
                with rawpy.imread(filepath) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
                    visible_image = raw.raw_image_visible.copy()
                    rgb = cv2.cvtColor(visible_image, cv2.COLOR_BAYER_RG2RGB) 
                    original_shape = rgb.shape[:2]
                    resized_rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA) 
                    normalized_rgb = resized_rgb.astype(np.float32) / 255.0

                    input_tensor = np.expand_dims(normalized_rgb, axis=0)
                    predicted_image = model.predict(input_tensor, verbose=0)[0]

                    predicted_image = cv2.resize(predicted_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    predicted_image = np.clip(predicted_image * 255, 0, 255).astype(np.uint8)

                output_filename = os.path.splitext(filename)[0] + "_predicted.jpg"
                output_path = os.path.join(output_dir, output_filename)
                iio.imwrite(output_path, predicted_image)
                logger.info(f"Image traitée : {filename} -> {output_filename}")
                processed_count += 1

            except Exception as e:
                logger.error(f"Erreur lors du traitement de {filename}: {e}")

    logger.info(f"Traitement terminé. {processed_count}/{num_images} images traitées.")


if __name__ == "__main__":
    predict_raw_images(MODEL_PATH, INPUT_RAW_DIR, OUTPUT_DIR)