# ============================================================
# CLEAN + TF2-COMPATIBLE MODELS (Generator / Discriminator / Siamese / ResUNet)
# - Silences TF "Executing op ..." spam
# - TF2-safe ReflectionPadding2D
# - ResBlock keeps SAME spatial size (reflect pad + valid conv) and enforces stride=1
# - No undefined globals; everything gets explicit input_shape
# ============================================================

import os

# ---------- SILENCE TF LOG SPAM (MUST be before importing tensorflow) ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
# If you don't want GPU at all, uncomment:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU,
    Dense, Dropout, Flatten, UpSampling2D, MaxPooling2D, Lambda, Add,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputSpec, Layer

from tqdm import tqdm

# Extra silencing (TF2)
tf.get_logger().setLevel("ERROR")
try:
    tf.debugging.set_log_device_placement(False)
except Exception:
    pass

# skimage version compatibility
try:
    from skimage.measure import compare_psnr, compare_ssim
except ImportError:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim

from sklearn.model_selection import train_test_split


# ============================================================
# Reflection padding (TF2-safe)
# ============================================================
def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """
    Pad the H and W dimensions of a 4D tensor using REFLECT padding.
    x: (B,H,W,C) for channels_last or (B,C,H,W) for channels_first
    padding: ((top,bottom),(left,right))
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format " + str(data_format))

    if (
        not isinstance(padding, (tuple, list))
        or len(padding) != 2
        or len(padding[0]) != 2
        or len(padding[1]) != 2
    ):
        raise ValueError(f"padding must be ((top,bottom),(left,right)). Got: {padding}")

    if data_format == "channels_first":
        pattern = [
            [0, 0],           # batch
            [0, 0],           # channels
            [padding[0][0], padding[0][1]],  # height
            [padding[1][0], padding[1][1]],  # width
        ]
    else:
        pattern = [
            [0, 0],           # batch
            [padding[0][0], padding[0][1]],  # height
            [padding[1][0], padding[1][1]],  # width
            [0, 0],           # channels
        ]

    return tf.pad(x, pattern, mode="REFLECT")


class ReflectionPadding2D(Layer):
    """
    Reflection-padding layer for 2D inputs.
    padding:
      - int -> symmetric padding for H and W
      - (h_pad, w_pad) -> symmetric per-dim
      - ((top,bottom),(left,right)) -> explicit
    """
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super().__init__(**kwargs)

        # Safe TF2-compatible data_format handling
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            if data_format not in {"channels_first", "channels_last"}:
                raise ValueError(f"Invalid data_format: {data_format}")
            self.data_format = data_format

        def normalize_tuple(value, n):
            if isinstance(value, int):
                return (value,) * n
            if isinstance(value, (tuple, list)) and len(value) == n:
                return tuple(value)
            raise ValueError(f"Invalid value {value}; expected int or tuple/list of len {n}")

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif isinstance(padding, (tuple, list)):
            if len(padding) != 2:
                raise ValueError(f"`padding` should have two elements. Found: {padding}")
            height_padding = normalize_tuple(padding[0], 2)
            width_padding = normalize_tuple(padding[1], 2)
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError(
                "`padding` should be int, (h_pad,w_pad), or ((top,bottom),(left,right)). "
                f"Found: {padding}"
            )

        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            rows = None if input_shape[2] is None else input_shape[2] + self.padding[0][0] + self.padding[0][1]
            cols = None if input_shape[3] is None else input_shape[3] + self.padding[1][0] + self.padding[1][1]
            return (input_shape[0], input_shape[1], rows, cols)
        else:
            rows = None if input_shape[1] is None else input_shape[1] + self.padding[0][0] + self.padding[0][1]
            cols = None if input_shape[2] is None else input_shape[2] + self.padding[1][0] + self.padding[1][1]
            return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs, padding=self.padding, data_format=self.data_format)

    def get_config(self):
        config = {"padding": self.padding, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}


# ============================================================
# ResNet block (CycleGAN-style) - SAFE (stride=1 only)
# ============================================================
def res_block(x_in, filters, kernel_size=(3, 3), use_dropout=False):
    """
    CycleGAN-style residual block.
    Keeps SAME spatial size by: reflect pad + valid conv.
    Stride is intentionally fixed to 1 to keep residual Add valid.
    """
    x = ReflectionPadding2D((1, 1))(x_in)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding="valid")(x)
    x = BatchNormalization()(x)

    return Add()([x_in, x])


# ============================================================
# Generator
# ============================================================
def generator_model(ngf=64, input_nc=1, output_nc=1, input_shape=(40, 60, 1), n_blocks_gen=16):
    inputs = Input(shape=input_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding="valid")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Downsampling
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2 ** i
        x = Conv2D(filters=ngf * mult * 2, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    # Res blocks
    mult = 2 ** n_downsampling
    for _ in range(n_blocks_gen):
        x = res_block(x, ngf * mult, use_dropout=True)

    # Upsampling
    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding="valid")(x)
    x = Activation("tanh")(x)

    # Skip connection (requires SAME spatial size)
    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z / 2.0)(outputs)

    return Model(inputs=inputs, outputs=outputs, name="Generator")


# ============================================================
# Discriminator
# ============================================================
def discriminator_model(ndf=64, input_shape=(40, 60, 1)):
    """
    A simple discriminator producing a single probability output.
    (Not strict PatchGAN output map, since it flattens + dense.)
    """
    n_layers = 3
    use_sigmoid = False

    inputs = Input(shape=input_shape)

    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding="same")(inputs)
    x = LeakyReLU(0.2)(x)

    # conv blocks
    for n in range(1, n_layers + 1):
        nf_mult = min(2 ** n, 8)
        stride = 2 if n < n_layers else 1
        x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding="same")(x)
    if use_sigmoid:
        x = Activation("sigmoid")(x)

    x = Flatten()(x)
    x = Dense(1024, activation="tanh")(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=x, name="Discriminator")


# ============================================================
# Generator+Discriminator wrapper (explicit input_shape)
# ============================================================
def generator_containing_discriminator_multiple_outputs(generator, discriminator, input_shape):
    inputs = Input(shape=input_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    return Model(inputs=inputs, outputs=[generated_images, outputs], name="G_D")


# ============================================================
# Siamese network
# ============================================================
def siamese_model(input_shape):
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    base = Sequential(name="SiameseBase")
    base.add(Conv2D(64, (10, 10), activation="relu", input_shape=input_shape))
    base.add(MaxPooling2D())
    base.add(Conv2D(128, (7, 7), activation="relu"))
    base.add(MaxPooling2D())
    base.add(Conv2D(128, (4, 4), activation="relu"))
    base.add(MaxPooling2D())
    base.add(Conv2D(256, (4, 4), activation="relu"))
    base.add(MaxPooling2D())
    base.add(Conv2D(256, (4, 4), activation="relu"))
    base.add(MaxPooling2D())
    base.add(Conv2D(256, (4, 4), activation="relu"))
    base.add(MaxPooling2D())
    base.add(Flatten())
    base.add(Dense(4096, activation="sigmoid"))

    encoded_l = base(left_input)
    encoded_r = base(right_input)

    l1 = Lambda(lambda t: K.abs(t[0] - t[1]))([encoded_l, encoded_r])
    prediction = Dense(1, activation="sigmoid")(l1)

    return Model(inputs=[left_input, right_input], outputs=prediction, name="Siamese")


# ============================================================
# Generator+Discriminator+Siamese wrapper (explicit input_shape)
# ============================================================
def generator_containing_siamese_multiple_inputs_outputs(generator, discriminator, siamese, input_shape):
    left_inputs = Input(shape=input_shape)
    right_inputs = Input(shape=input_shape)

    generated_images = generator(left_inputs)
    discriminator_outputs = discriminator(generated_images)
    siamese_outputs = siamese([generated_images, right_inputs])

    return Model(
        inputs=[left_inputs, right_inputs],
        outputs=[generated_images, discriminator_outputs, siamese_outputs],
        name="G_D_Siamese"
    )


# ============================================================
# ResUNet (clean + TF2-safe + axis-safe concat)
# ============================================================
def _concat_axis():
    return 1 if K.image_data_format() == "channels_first" else 3


def unet_res_block(x, nb_filters, kernel_size=(3, 3), strides=((1, 1), (1, 1)), use_dropout=False):
    res_path = Conv2D(filters=nb_filters[0], kernel_size=kernel_size, padding="same", strides=strides[0])(x)
    res_path = BatchNormalization()(res_path)
    res_path = Activation("relu")(res_path)

    if use_dropout:
        res_path = Dropout(0.5)(res_path)

    res_path = Conv2D(filters=nb_filters[1], kernel_size=kernel_size, padding="same", strides=strides[1])(res_path)
    res_path = BatchNormalization()(res_path)

    shortcut = Conv2D(filters=nb_filters[1], kernel_size=(1, 1), padding="same", strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    out = Add()([shortcut, res_path])
    out = Activation("relu")(out)
    return out


def unet_encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding="same", strides=(1, 1))(x)
    main_path = BatchNormalization()(main_path)
    main_path = Activation("relu")(main_path)

    main_path = Conv2D(filters=64, kernel_size=(3, 3), padding="same", strides=(1, 1))(main_path)
    main_path = BatchNormalization()(main_path)

    shortcut = Conv2D(filters=64, kernel_size=(1, 1), padding="same", strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = Add()([shortcut, main_path])
    main_path = Activation("relu")(main_path)

    to_decoder.append(main_path)

    main_path = unet_res_block(main_path, nb_filters=[128, 128], strides=((2, 2), (1, 1)))
    to_decoder.append(main_path)

    main_path = unet_res_block(main_path, nb_filters=[256, 256], strides=((2, 2), (1, 1)))
    to_decoder.append(main_path)

    return to_decoder


def unet_decoder(x, from_encoder):
    axis = _concat_axis()

    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = Concatenate(axis=axis)([main_path, from_encoder[2]])
    main_path = unet_res_block(main_path, nb_filters=[256, 256], strides=((1, 1), (1, 1)))

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = Concatenate(axis=axis)([main_path, from_encoder[1]])
    main_path = unet_res_block(main_path, nb_filters=[128, 128], strides=((1, 1), (1, 1)))

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = Concatenate(axis=axis)([main_path, from_encoder[0]])
    main_path = unet_res_block(main_path, nb_filters=[64, 64], strides=((1, 1), (1, 1)))

    return main_path


def build_res_unet(input_shape=(40, 60, 1)):
    inputs = Input(shape=input_shape)
    to_decoder = unet_encoder(inputs)

    path = unet_res_block(to_decoder[2], nb_filters=[512, 512], strides=((2, 2), (1, 1)))
    path = unet_decoder(path, from_encoder=to_decoder)

    outputs = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid", padding="same")(path)
    return Model(inputs=inputs, outputs=outputs, name="ResUNet")