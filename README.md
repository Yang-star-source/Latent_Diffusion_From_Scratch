# Latent Diffusion From Scratch: Unconditional and Text-to-Image (T2I) Generation.

|![Cat Grid 1](images/sample_cat1.png)|![Cat Grid 2](images/ImageSampling800.png)|
| :---: | :---: |
| **Unconditional Image Generation** | **Prompt : A cute orange cat with green eyes** |

## Architecture Overview
There is 3 models to be trained , Variational AutoEncoders (VAE) , Unconditional Diffusion U-Net and Text-Conditioned Diffusion U-Net.
Both Diffusion U-Net use the same VAE during inference mode.

<img src="images/introduction1.png" width="60%">

| **Unconditional** | **Text-Conditioned** |
| :---: | :---: |
| ![Uncondition](images/introduction2.png) | ![condition](images/introduction3.png) |
| $$L = \mathbb{E}_{z, y, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|^2 \right]$$ | $$L = \mathbb{E}_{z, y, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

## More Details Architecture Explanation
Need flowchart for clearer explanation ?
[**Click here to browse the architecture flowcharts**](FlowChart/)

I have provided the source file for anyone who wants to study the architecture deeply or modify the diagrams.

* **[Download Source File (.drawio)](FlowChart.drawio)**
  
*(Note: To open this file, go to [app.diagrams.net](https://app.diagrams.net/) in your browser and drag-and-drop the file onto the canvas.)*

To view my draft and notes when studying Latent Diffusion :

[View the Draft and Notes](https://1drv.ms/o/c/eae736d6f2d45eed/IgAV2N1tMUu1QaddKr_KNX88AeZ-4xvzktsisAO-h5fDtRE?e=DPCh6h)

*(Note: The codes wrote inside notes might be differ with the actual implementing codes)*

## Pre-trained Models
You can download the trained weights from Google Drive:

* **[Download Pre-trained Weights Folder](https://drive.google.com/drive/folders/1EbwM13No2i44OCQbxzMkeORkeEqaT9IU?usp=sharing)**
  * `checkpoint.pth` (VAE Autoencoder)
  * `unet_epoch_500.pth` (Diffusion Model)

**Setup:**
1. Download both files.
2. Place them inside your project folder (or upload to Colab).


## Codes Implementation
For training code (Just View) :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yang-star-source/Latent_Diffusion_From_Scratch/blob/main/Latent_Diffusion_From_Scratch.ipynb)

For Inference Mode (One Click Run):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yang-star-source/Latent_Diffusion_From_Scratch/blob/main/Latent_Diffusion_From_Scratch_Inference.ipynb)

## Dataset 
```bibtex
@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

[Dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

[CATS ONLY used in this training](https://huggingface.co/datasets/ziyang06315/cats_images_dataset/tree/main)

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
