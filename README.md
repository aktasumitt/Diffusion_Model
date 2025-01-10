# Diffusion Model

## Introduction:
In this project, I aimed to create a Diffusion model with Cifar10 dataset for Create image from random noise

## Dataset:
- I used the Cifar10 and Mnist datasets for this project, which consists of 10 labels with total 60k images on train and 10k images on test.
- size of images are (32x32x3), i transform to (64x64x3)

## Model:
- Diffusion modeling essentially consists of two main stages. The first stage is the Diffusion stage, where we add noise to the image at regular intervals (using a cosine distribution), gradually corrupting the image until it becomes entirely noise. The second stage involves using a model (such as U-net) to learn from these noises and attempt to reconstruct the original image..

#### Diffusion Stage :
- (All mathematical names mentioned here are derived from the formulas in the article).
- Firstly, we create a "beta" tensor with 1000 steps at equal intervals.
-  Then, we create an "alpha" tensor using the formula (alpha=1-beta). These tensors will help us generate noises with a normal distribution.
-  Next, by applying cumulative product to the alpha tensor, we create the "alpha hat" tensor. This ensures that all noise steps are contained within a single tensor because the noise will increase at each step within the tensor.
-  Using the formula (sqrt(alpha)*x+sqrt(alpha_hat)*epsilon) where epsilon represents normal distribution random data like "torch.randn", we apply noise to x at each step. Since the noise increases at each step, the image will become more corrupted with each step.
- Using the model, we will attempt to gradually reach the real images by predicting the noise within the image at each step, using the formula (new_x = 1 / torch.sqrt(alpha)* (x- ((1-alpha)) / (torch.sqrt(1-alpha_hat))*predicted_noise) + torch.sqrt(beta)*noise). 
- Essentially, we are trying to predict the noise within the image rather than the image itself.

#### Model Stage :
- The model here essentially utilizes a U-Net architecture, using the downsample system within the encoder to compress input dimensions into a latent space, and then using the decoder to expand this latent space back into high-dimensional images.
-  In this model, we will provide noisy images to the model and ask it to predict the noise within the image for that step.
-  We will then subtract this noise and apply the same process again, progressing until we eventually find the image without noise.
-  Additionally, we scale the time values with embeddings and feed them to the model during downsampling and upsampling. We then combine these time values with the image output so that the model knows at which time value it is attempting to predict output.
-  Since we are using a conditional model, we embed the labels as well, adding them to the t dimension.

## Train:
- During training, we apply these steps individually for each batch of images, and eventually generate images through sampling.
- We teach the model to remove noise from the image, essentially teaching it to generate images from noise because at the final stage where we reach the maximum noise level, there is no trace of the original image left.
- We use the 'Adam' optimizer and Mean Square Error Loss function during training.
