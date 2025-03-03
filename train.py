from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization, Activation

def build_unet(image_size=(512, 512, 3)):
    inputs = Input(shape=image_size)

    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(128, (3, 3), padding="same", strides=2)(x)  
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    outputs = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)  # Sortie finale normalisée

    model = Model(inputs, outputs, name="PhotoEnhancer")
    return model

model = build_unet()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
