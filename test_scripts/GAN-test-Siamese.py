
# coding: utf-8

# In[ ]:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all, 1=filter INFO, 2=filter INFO+WARNING, 3=only ERROR

from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import os

print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

import tensorflow as tf

# Enable dynamic GPU memory growth (TF2 way)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Optional: log device placement
tf.debugging.set_log_device_placement(True)


# In[18]:

import imageio
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#import keras

from tqdm import tqdm
from itertools import chain
import datetime
from skimage.io import imread, imshow, concatenate_images

from skimage.measure import compare_psnr, compare_ssim


from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

#import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
#import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout


from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
os.chdir("../")
cwd = os.getcwd()

from Models.models import *
input_shape = np.array([448,896,1])

ngf = 64
ndf = 64 
n_blocks_gen = 16

#Give path to the pretrained model
now = datetime.datetime.now()
saved_dir = os.path.join('GAN/', '{}{}'.format(now.month, now.day))
model_path = os.path.join(cwd, "saved_model/")
print(model_path)

# In[24]:


im_width = 896
im_height = 448
border = 5
path_train = cwd + '/dataset/train/'
path_test = cwd + '/dataset/test/'

def get_data(path, train=True):
    ids = next(os.walk(path + "raw"))[2]

    
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load raw
        img = load_img(path + 'raw/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (448,896, 1), mode='constant', preserve_range=True)

        # Load average
        if train:
            average = img_to_array(load_img(path + 'average/' + id_, grayscale=True))
            average = resize(average, (448, 896, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = average / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
x_test, y_test = get_data(path_test, train=True)


# In[ ]:


import numpy as np
# from PIL import Image
import click
import scipy.misc
import statistics
# from scipy import stats

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 


# ===========================
# ADDITIONAL METRICS SUPPORT
# ===========================
import torch
import lpips

lpips_model = lpips.LPIPS(net='alex')
lpips_model.eval()


def test(batch_size):
    max_val = 0
    ids = next(os.walk(model_path))[2]
    print("Files inside saved_model:", ids)

    g = generator_model(
        ngf=ngf,
        input_nc=input_shape[2],
        output_nc=input_shape[2],
        input_shape=input_shape,
        n_blocks_gen=n_blocks_gen
    )

    for temp in ids:
        if "generator.h5" in temp:

            # Metric containers
            p = []
            q = []
            r = []
            mae_list = []
            mse_list = []
            lpips_list = []

            temp = model_path + "/" + temp
            g.load_weights(temp)

            generated_images = g.predict(x=x_test, batch_size=batch_size)

            for i in range(generated_images.shape[0]):
                y = y_test[i, :, :, :]
                x = x_test[i, :, :, :]
                img = generated_images[i, :, :, :]

                output = np.concatenate((y, x, img), axis=1)

                # --------------------
                # PSNR / SSIM
                # --------------------
                p.append(compare_psnr(img[:,:,0], y[:,:,0]))
                q.append(compare_ssim(img[:,:,0], y[:,:,0]))
                r.append(signaltonoise(img[:,:,0], axis=0, ddof=0))

                # --------------------
                # MAE
                # --------------------
                mae = np.mean(np.abs(img[:,:,0] - y[:,:,0]))
                mae_list.append(mae)

                # --------------------
                # MSE
                # --------------------
                mse = np.mean((img[:,:,0] - y[:,:,0]) ** 2)
                mse_list.append(mse)

                # --------------------
                # LPIPS
                # --------------------
                img_rgb = np.repeat(img, 3, axis=2)
                y_rgb = np.repeat(y, 3, axis=2)

                img_t = torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).float()
                y_t = torch.tensor(y_rgb).permute(2,0,1).unsqueeze(0).float()

                img_t = img_t * 2 - 1
                y_t = y_t * 2 - 1

                lpips_value = lpips_model(img_t, y_t)
                lpips_list.append(lpips_value.item())

            # ========================
            # Compute Means
            # ========================
            psnr_mean = np.mean(p)
            ssim_mean = np.mean(q)
            mae_mean = np.mean(mae_list)
            mse_mean = np.mean(mse_list)
            lpips_mean = np.mean(lpips_list)

            if psnr_mean > max_val:
                max_val = psnr_mean

                for i in range(generated_images.shape[0]):
                    y = y_test[i, :, :, :]
                    x = x_test[i, :, :, :]
                    img = generated_images[i, :, :, :]

                    imageio.imwrite(
                        cwd + '/GAN/Results/results{}.png'.format(i+1),
                        (img[:,:,0] * 255).astype(np.uint8)
                    )

            # ========================
            # Logging
            # ========================
            with open(cwd + '/GAN/Results/logs.txt', "a") as f:
                f.write("generator is " + str(temp) + "\n")
                f.write("psnr = " + str(psnr_mean) + "\n")
                f.write("ssim = " + str(ssim_mean) + "\n")
                f.write("mae = " + str(mae_mean) + "\n")
                f.write("mse = " + str(mse_mean) + "\n")
                f.write("lpips = " + str(lpips_mean) + "\n")
                f.write("current max PSNR = " + str(max_val) + "\n\n")

            # ========================
            # Print Results
            # ========================
            print("generator is", temp)
            print("PSNR =", psnr_mean)
            print("SSIM =", ssim_mean)
            print("MAE  =", mae_mean)
            print("MSE  =", mse_mean)
            print("LPIPS=", lpips_mean)
            print("current max PSNR =", max_val)


test(1)