# coding: utf-8

import os
import datetime
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import statistics

import tensorflow as tf

import h5py

import h5py

path = "/mnt/e/PDRA/SiameseGAN-master/saved_model/generator.h5"

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print("Attributes:", list(f.attrs.keys()))

# ============================================================
# GAN-test_Siamese_UNET.py  (TF2 / tf.keras compatible)
# - Uses tf.keras ONLY (no standalone "keras")
# - Loads generator.h5 as WEIGHTS (your file looks like weights-only)
# - Fixes paths + missing imports + tqdm usage in terminal
# ============================================================

import os
import datetime
import statistics
import numpy as np
import imageio

# --------------------------
# GPU
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
print("TensorFlow:", tf.__version__)

# --------------------------
# External libs
# --------------------------
from tqdm import tqdm
from skimage.transform import resize

# skimage API changed across versions; handle both old and new
try:
    # old (skimage <=0.17/0.18)
    from skimage.measure import compare_psnr as psnr_fn
    from skimage.measure import compare_ssim as ssim_fn
except Exception:
    # new (skimage >=0.19)
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --------------------------
# Project paths
# --------------------------
# You run this inside: .../SiameseGAN-master/test_scripts
# so cwd should become .../SiameseGAN-master
os.chdir("..")
cwd = os.getcwd()

# IMPORTANT:
# Your Models/models.py MUST import from tensorflow.keras (NOT "keras")
# e.g. "from tensorflow.keras.models import Model"
from Models.models import build_res_unet

# --------------------------
# Settings
# --------------------------

input_shape = (1, 448, 896)
im_height, im_width = 448, 896

path_test = os.path.join(cwd, "dataset", "test")

# 1) optional location (if you copy generator.h5 here)
generator_h5_path = os.path.join(cwd, "GAN", "generator.h5")

# 2) your real location (you said saved_model folder)
generator_h5_path_alt = os.path.join(cwd, "saved_model", "generator.h5")


# --------------------------
# Data loader
# --------------------------
def get_data(path, train=True):
    raw_dir = os.path.join(path, "raw")
    avg_dir = os.path.join(path, "average")

    ids = next(os.walk(raw_dir))[2]
    ids.sort()

    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32) if train else None

    print("Getting and resizing images ...")
    for n, id_ in tqdm(list(enumerate(ids)), total=len(ids)):
        # RAW
        img = load_img(os.path.join(raw_dir, id_), color_mode="grayscale")
        x_img = img_to_array(img)
        x_img = resize(
            x_img,
            (im_height, im_width, 1),
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )

        X[n, ..., 0] = x_img.squeeze() / 255.0

        # AVERAGE (GT)
        if train:
            avg_img = load_img(os.path.join(avg_dir, id_), color_mode="grayscale")
            avg_arr = img_to_array(avg_img)
            avg_arr = resize(
                avg_arr,
                (im_height, im_width, 1),
                mode="constant",
                preserve_range=True,
                anti_aliasing=False,
            )
            y[n, ..., 0] = avg_arr.squeeze() / 255.0

    print("Done!")
    return (X, y) if train else X


x_test, y_test = get_data(path_test, train=True)


# --------------------------
# Helpers
# --------------------------
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# --------------------------
# Test
# --------------------------
def test(batch_size=1):
    # Build model architecture
    g = build_res_unet(input_shape=input_shape)

    # ---- Load generator weights (weights-only .h5)
    used_path = None
    if os.path.exists(generator_h5_path):
        used_path = generator_h5_path
    elif os.path.exists(generator_h5_path_alt):
        used_path = generator_h5_path_alt
    else:
        raise FileNotFoundError(
            "Could not find generator.h5 in:\n"
            f"  1) {generator_h5_path}\n"
            f"  2) {generator_h5_path_alt}\n"
            "Place generator.h5 in one of these paths, or update the path."
        )

    # by_name=True helps if layer naming matches; if it still fails, set by_name=False
    try:
        g.load_weights(used_path, by_name=True)
    except Exception as e:
        print("by_name=True failed, retrying with by_name=False ...")
        g.load_weights(used_path, by_name=False)

    print("Loaded weights from:", used_path)

    generated_images = g.predict(x=x_test, batch_size=batch_size)

    p, q, r = [], [], []
    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, 0]
        img = generated_images[i, :, :, 0]

        p.append(psnr_fn(y, img, data_range=1.0) if "data_range" in psnr_fn.__code__.co_varnames else psnr_fn(img, y))
        q.append(ssim_fn(y, img, data_range=1.0) if "data_range" in ssim_fn.__code__.co_varnames else ssim_fn(img, y))
        r.append(signaltonoise(img, axis=0, ddof=0))

    psnr_mean = float(np.mean(p))

    results_dir = os.path.join(cwd, "GAN", "Results")
    os.makedirs(results_dir, exist_ok=True)

    # Save outputs
    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, 0]
        x = x_test[i, :, :, 0]
        img = generated_images[i, :, :, 0]

        imageio.imwrite(os.path.join(results_dir, f"results{i+1}.png"), (img * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(results_dir, f"raw{i+1}.png"), (x * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(results_dir, f"average{i+1}.png"), (y * 255).astype(np.uint8))

    # Log
    log_path = os.path.join(results_dir, "logs.txt")
    with open(log_path, "a") as f:
        f.write("generator file = {}\n".format(used_path))
        f.write("psnr mean = {}\n".format(psnr_mean))
        if len(p) > 1:
            f.write("psnr std  = {}\n".format(statistics.stdev(p)))
        f.write("ssim mean = {}\n".format(float(np.mean(q))))
        f.write("\n")

    print("PSNR mean:", psnr_mean)
    if len(p) > 1:
        print("PSNR std:", statistics.stdev(p))
    print("SSIM mean:", float(np.mean(q)))


test(1)