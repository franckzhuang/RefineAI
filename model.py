import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Add, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob
import os
from utils import load_image, calculate_psnr, calculate_ssim, preprocess_image, logger
import time
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

BATCH_SIZE = 2 
INITIAL_IMAGE_SIZE = (128, 128)
IMAGE_SIZES = [(128, 128), (256, 256), (512, 512), (1024, 1024)]  
EPOCHS_PER_SIZE = [20, 20, 30, 30]  
LEARNING_RATE = 1e-4
L2_REG = 1e-4
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
VAL_RAW_DIR = "data/val/raw"
VAL_PROCESSED_DIR = "data/val/processed"
MODEL_SAVE_PATH = "saved_model/my_model"


def build_model(input_shape=(None, None, 3)):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(inputs)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Dropout(0.2)(conv2)

    conv3 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(conv2)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_REG))(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(0.2)(conv3)

    outputs = Conv2D(3, (1, 1), activation='sigmoid', dtype='float32')(conv3)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def augment_data(raw, processed):
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    raw = tf.image.rot90(raw, k=k)
    processed = tf.image.rot90(processed, k=k)

    if tf.random.uniform(shape=[]) > 0.5:
        raw = tf.image.flip_left_right(raw)
        processed = tf.image.flip_left_right(processed)
    if tf.random.uniform(shape=[]) > 0.5:
        raw = tf.image.flip_up_down(raw)
        processed = tf.image.flip_up_down(processed)

    raw = tf.image.random_brightness(raw, max_delta=0.1)
    processed = tf.image.random_brightness(processed, max_delta=0.1)

    raw = tf.image.random_contrast(raw, lower=0.8, upper=1.2)
    processed = tf.image.random_contrast(processed, lower=0.9, upper=1.1)

    raw = tf.image.random_saturation(raw, lower=0.8, upper=1.2)
    raw = tf.image.random_hue(raw, max_delta=0.05)

    raw = tf.clip_by_value(raw, 0.0, 1.0)
    processed = tf.clip_by_value(processed, 0.0, 1.0)

    return raw, processed


def create_dataset(raw_dir, processed_dir, image_size, batch_size, augment=False):
    raw_paths = sorted(glob(os.path.join(raw_dir, "*.CR2")))
    processed_paths = sorted(glob(os.path.join(processed_dir, "*.jpg")))

    if len(raw_paths) != len(processed_paths):
        logger.error("Le nombre d'images RAW et JPEG ne correspond pas.")
        raise ValueError("Le nombre d'images RAW et JPEG ne correspond pas.")
    if not raw_paths:
        logger.error("Aucune image RAW trouvée.")
        raise ValueError("Aucune image RAW trouvée.")
    if not processed_paths:
        logger.error("Aucune image JPEG trouvée.")
        raise ValueError("Aucune image JPEG trouvée.")

    num_images = len(raw_paths)
    logger.info(f"Chargement de {num_images} paires d'images...")

    def load_and_preprocess(raw_path_tensor, processed_path_tensor):
        raw_path = raw_path_tensor.numpy().decode()
        processed_path = processed_path_tensor.numpy().decode()
        logger.info(f"Chargement de : {os.path.basename(raw_path)} et {os.path.basename(processed_path)}")
        raw = load_image(raw_path, image_size)
        processed = load_image(processed_path, image_size)
        if raw is None or processed is None:
            logger.warning(f"Paire d'images ignorée : {os.path.basename(raw_path)}, {os.path.basename(processed_path)}")
            return tf.zeros((0,) + image_size + (3,), dtype=tf.float32), tf.zeros((0,) + image_size + (3,), dtype=tf.float32)
        return preprocess_image(raw), preprocess_image(processed)

    def filter_empty_images(raw, processed):
        return tf.reduce_sum(tf.shape(raw)) > 0

    dataset = tf.data.Dataset.from_tensor_slices((raw_paths, processed_paths))
    dataset = dataset.map(lambda x, y: tf.py_function(load_and_preprocess, [x, y], [tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(filter_empty_images)

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()
    return dataset


def train_model(model, raw_dir, processed_dir, val_raw_dir, val_processed_dir, batch_size, image_sizes, epochs_per_size, learning_rate):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=[calculate_psnr, calculate_ssim])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH + "_best",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv", append=True),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: logger.info(
                f"Epoch {epoch+1} - loss: {logs['loss']:.4f} - val_loss: {logs.get('val_loss', 'N/A') if 'val_loss' in logs else 'N/A'}:{'.4f' if 'val_loss' in logs else ''} - PSNR: {logs['calculate_psnr']:.2f} - SSIM: {logs['calculate_ssim']:.4f}"
            )
        ),
    ]

    total_epochs = 0
    for i, size in enumerate(image_sizes):
        logger.info(f"Entraînement avec une taille d'image de {size}...")
        train_dataset = create_dataset(raw_dir, processed_dir, size, batch_size, augment=True)
        val_dataset = create_dataset(val_raw_dir, val_processed_dir, size, batch_size, augment=False)

        logger.info("Début de l'entraînement...")
        start_time = time.time()
        try:
            history = model.fit(
                train_dataset,
                epochs=epochs_per_size[i],
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=0,
                initial_epoch=total_epochs
            )
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Erreur de mémoire lors de l'entraînement avec la taille {size}.  Passage à la taille suivante.")
            continue
        finally:
            end_time = time.time()
            logger.info(f"Entraînement terminé en {end_time - start_time:.2f} secondes pour la taille {size}.")
            total_epochs += epochs_per_size[i]

    return model

def evaluate_model(model, test_dataset):
    logger.info("Évaluation du modèle...")
    loss, psnr, ssim = model.evaluate(test_dataset, verbose=0)
    logger.info(f"Loss (MSE): {loss:.4f}")
    logger.info(f"PSNR (Moyenne): {psnr:.2f} dB")
    logger.info(f"SSIM (Moyenne): {ssim:.4f}")


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU disponible: {gpus}")
            policy = Policy('mixed_float16')
            set_global_policy(policy)
            logger.info('Mixed precision activée.')
        except RuntimeError as e:
            logger.error(f"Erreur lors de la configuration du GPU: {e}")
    else:
        logger.warning("Aucun GPU détecté. L'entraînement se fera sur CPU.")

    test_dataset = create_dataset(RAW_DIR, PROCESSED_DIR, INITIAL_IMAGE_SIZE, BATCH_SIZE * 4)

    model = build_model(input_shape=(INITIAL_IMAGE_SIZE[0], INITIAL_IMAGE_SIZE[1], 3))
    model.summary(print_fn=logger.info)

    model = train_model(model, RAW_DIR, PROCESSED_DIR, VAL_RAW_DIR, VAL_PROCESSED_DIR, BATCH_SIZE, IMAGE_SIZES, EPOCHS_PER_SIZE, LEARNING_RATE)

    evaluate_model(model, test_dataset)

    model.save(MODEL_SAVE_PATH)
    logger.info(f"Modèle sauvegardé dans {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()