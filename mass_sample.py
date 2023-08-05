import torch
from Unet import Unet
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sys

def p_sample(model, x, t, t_index):
    with torch.no_grad():
        model_mean = model(x,t)
    # if t_index ==1:
    #     return model_mean
    # else:
    #     posterior_variance_t = extract(posterior_variance, t-1, x.shape)
    #     noise = torch.randn_like(x)
    #     return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
        return model_mean

def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(1, timesteps+1)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.detach().cpu().numpy())
    return imgs

def sample(model, image_size=28, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

timesteps = 300
image_size = 28
channels = 1
batch_size = 64

device = torch.device('cuda:0')
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4)
)
model.to(device)

checkpoint = torch.load(sys.argv[1])
model.load_state_dict(checkpoint['net'])

# samples = sample(model, image_size=image_size, batch_size=1, channels=channels)


# plot the actual results
x = int(sys.argv[2])
y = int(sys.argv[3])

fig, axs = plt.subplots(x,y)

for i in range(x):
    for e in range(y):
        samples = sample(model, image_size=image_size, batch_size=1, channels=channels)
        axs[i,e].imshow(samples[-1][0].reshape(image_size, image_size, channels), cmap="gray")
        axs[i,e].axis("off")
plt.show()


