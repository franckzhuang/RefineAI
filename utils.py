import rawpy
import imageio.v3 as iio
import numpy as np
import cv2
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

def load_image(image_path, target_size=(256, 256)):
    try:
        if image_path.lower().endswith('.cr2'):
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
                visible_image = raw.raw_image_visible.copy()
                rgb = cv2.cvtColor(visible_image, cv2.COLOR_BAYER_RG2RGB)
                rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
                return rgb.astype(np.float32) / 255.0
        else:
            image = iio.imread(image_path)
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            return image.astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {os.path.basename(image_path)}: {e}")
        return None

def calculate_psnr(img1, img2):
    return tf.image.psnr(img1, img2, max_val=1.0)

def calculate_ssim(img1, img2):
    return tf.image.ssim(img1, img2, max_val=1.0)

def preprocess_image(image):
    return image