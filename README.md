# Latent Diffusion From Scratch

An implementation of a Latent Diffusion Model (LDM) built from scratch using PyTorch.

<p>
  <img src="images/sample_cat1.png" width="600" title="Generated Results">
</p>

*Generated samples using the custom trained model.*

## Architecture Overview
There is 2 models to be trained , Variational AutoEncoders (VAE) and Diffusion U-Net

![An Introduction of Model Architecture](images/introduction.png)

## Pre-trained Models
You can download the trained weights from Google Drive:

* **[Download Pre-trained Weights Folder](https://drive.google.com/drive/folders/1EbwM13No2i44OCQbxzMkeORkeEqaT9IU?usp=sharing)**
  * `checkpoint.pth` (VAE Autoencoder)
  * `unet_epoch_500.pth` (Diffusion Model)

**Setup:**
1. Download both files.
2. Place them inside your project folder (or upload to Colab).


## Codes Implementation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yang-star-source/Latent_Diffusion_From_Scratch/blob/main/Latent_Diffusion_From_Scratch.ipynb)

## Dataset 
[Dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## Study Resources

### Papers
* **Denoising Diffusion Probabilistic Models** - [Ho et al. (2020)](https://arxiv.org/abs/2006.11239)
* **Attention Is All You Need** - [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
* **High-Resolution Image Synthesis with Latent Diffusion Models** - [Rombach et al. (2021)](https://arxiv.org/abs/2112.10752)

### Articles
* **Diffusion Model from Scratch** - [Hugging Face](https://huggingface.co/learn/diffusion-course/unit1/3)
* **KL Divergence** - [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/kullback-leibler-divergence/)

### Videos
* [Diffusion Models: DDPM | Generative AI Animated](https://www.youtube.com/watch?v=EhndHhIvWWw) – Excellent visualization of the forward/reverse process.
* [The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I) (Artem Kirsanov) – Essential for understanding the entropy/KL-divergence math.
* [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention) (3Blue1Brown) – Visual breakdown of the Attention Mechanism.
