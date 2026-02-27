# sml-iisc-SiameseGAN
Updated code for SiameseGAN under TensorFlow 2.x with fixed testing pipeline and extended evaluation metrics (PSNR, SSIM, MAE, MSE, LPIPS) for OCT image denoising.


* ###### SiameseGAN – Updated Implementation (TF2 Compatible) 

## Acknowledgment

This code is based on the original SiameseGAN implementation proposed in:
https://github.com/sml-iisc/SiameseGAN/tree/master 


# While using the original implementation, multiple compatibility issues were encountered — particularly within GAN-test-Siamese.py, due to:

* Deprecated TensorFlow/Keras APIs

* Deprecated skimage.measure metrics

* Image saving dtype conflicts

* Model loading inconsistencies

* Saved model name


# This repository contains:

✅ Fixed TensorFlow 2.x compatibility

✅ Corrected testing script

✅ Proper image saving (uint8 conversion)

✅ Added evaluation metrics:

* MAE

* MSE

* LPIPS (Perceptual metric)

# ########################################################################################## 
# Environment Setup

We use Python 3.8

# Create environment:

conda create -n gan_tf2 python=3.8 -y
conda activate gan_tf2

# Install required packages:

pip install tensorflow==2.4.1
pip install numpy==1.19.5
pip install scipy==1.5.4
pip install scikit-image==0.17.2
pip install scikit-learn==0.24.2
pip install matplotlib==3.3.4
pip install pandas==1.1.5
pip install imageio
pip install tqdm
pip install pillow
pip install torch
pip install lpips

# ########################################################################################## 
# 📂 Dataset

The existing code uses the SDOCT dataset:

28 total images

10 for training

18 for testing

# Directory structure:

* dataset/train/
* dataset/test/
* dataset/test/real/

# ########################################################################################## 
# Training

Navigate to training scripts:

cd /path_to_project/train_scripts

Train models:

python GAN-SIAMESE.py
python GAN-ResNet.py
python GAN-SIAMESE-UNET.py

By default, models are trained with:

Combined perceptual loss

MS-SSIM loss

# ########################################################################################## 
# Testing

Navigate to test scripts:

cd /path_to_project/test_scripts

Run testing:

* python GAN-test-Siamese.py

# This will compute:

* PSNR

* SSIM

* MAE

* MSE

* LPIPS

# Output images are saved to:

GAN/Results/

# ########################################################################################## 
# To compute MSR and CNR:

python msr_cnr.py
python msr_cnr2.py

# ########################################################################################## 
# Model Definitions

All model architectures are defined in:

Models/models.py

# ########################################################################################## 
Pretrained Models

Pretrained model available:

* saved_model/generator.h5

This is the best-performing SiameseGAN (MS-SSIM) model.

Other models available:

unet.h5

wgan_resenet.h5

# ########################################################################################## 
# Image Denoising Script

To denoise custom images:

cd test_scripts
python image_denoise /absolute/path/to/images/

Example:

python image_denoise /home/user/SiameseGAN-master/dataset/test/real/



### Reference Work
Links for the models which are used for comaparison.

https://github.com/sml-iisc/SiameseGAN/tree/master 

