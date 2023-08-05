# diffusion_model
Custom Diffusion Model. The forward process allows for linear, quadratic, sigmoid and cosine beta schedules. The backwards process uses U-Net to denoise.
Trained on the mnist dataset. 
<br>
The Diffusion Model: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ 

## Dependencies
- Pytorch
- tqdm
- datasets
- numpy
- matplotlib
- einops

## Included Modules
- Unet : The Unet image segementation model for 28 x 28 images
- forward_process : creates a forward process class that defines a forward process (number of timesteps and beta schedule)
- beta_schedule : various beta schedules to use for the forward process
- load_data : loads the mnist dataset as a dataloader
- mass_sample : samples the UNet denoising process creating a matplotlib figure with customizable size
- sample : samples the Unet once
- timeshow_sample : shows the denoising process from random noise to the final image
- train_unet : trains the unet for a given number of epochs
