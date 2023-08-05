import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F

######## beta schedule ######
import beta_schedule

######## Functions ##############
def extract(a, t, x_shape):
	'''
		this function should extract the timestep t from a and return that timestep in x_shape.
		*From Hanjing 

		a : array-like (one dimension)
		t : int
		x_shape : (int, int)
	'''
	batch_size = t.shape[0]
	out = a.gather(-1, t.cpu())
	return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

######## classes ################

class Forward(object):
	""" 
		Class to handle the forward process of the diffusion model
		
	Parameters
	----------
		beta_method : (int) -> array-like
		timesteps : int

	"""
	def __init__(self, beta_method, timesteps):
		super(Forward, self).__init__()
		self.beta_schedule = beta_method
		self.timesteps = timesteps

		# create betas according to schedule
		self.betas = self.beta_schedule(timesteps=timesteps)

		# These are generated according to this formula: q(x_t | x_0) = N(sqrt(a_t) * x_0, (1 - a_t)I)
		# generate alphas from betas
		self.alphas = 1 - self.betas
		self.alphas_bar = torch.cumprod(self.alphas, axis=0)
		self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)

		# generate variances
		self.variances = 1 - self.alphas_bar
		self.std_dev = torch.sqrt(self.variances)

	def forward_process(self, x_0, t, noise=None):
		'''
			Given the initial image and the timestep, return the image at time t in the forward process
		'''

		#create random noise if no noise is given
		if noise is None:
			noise = torch.randn_like(x_0)

		# extract the alphas and the variance
		sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, x_0.shape)
		std_dev_t = extract(self.std_dev, t, x_0.shape)

		# sqrt_alphas_bar_t = self.sqrt_alphas_bar.gather(-1, t.cpu()).numpy()[0]
		# std_dev_t = self.std_dev.gather(-1, t.cpu()).numpy()[0]

		out = sqrt_alphas_bar_t * x_0 + std_dev_t * noise
		# print("got:",out.shape)

		return out

	def get_next_step(self, x_t, t, noise=None):
		'''
			Given the initial image and the timestep, return the image at time t in the forward process
		'''

		#create random noise if no noise is given
		if noise is None:
			noise = torch.randn_like(x_t)

		beta_t = extract(self.betas, t-1, x_t.shape)
		one_minus_beta_t = 1-beta_t

		return torch.sqrt(one_minus_beta_t)* x_t + torch.sqrt(beta_t) * noise


if __name__ == "__main__":
	######## Parameters for the forward process ########
	timesteps = 300
	schedule_used = beta_schedule.linear_beta_schedule
	forward = Forward(schedule_used, timesteps)

	import load_data

	# dataloader = load_data.get_dataloader(0)

	# batch = next(iter(dataloader))

	# image = batch['pixel_values'][[0]]

	# plot([forward.forward_process(image, torch.tensor([t])).numpy()[0,0] for t in [i for i in range(0,timesteps, int(timesteps / 6))]])
	# plt.show()

	loader = load_data.get_dataloader()
	batch = next(iter(loader))['pixel_values']

	colored = load_data.get_mnist_batch(batch)

	# plot([forward.forward_process(torch.tensor(colored[0]), torch.tensor([t])).numpy()[0,0] for t in [i for i in range(0,timesteps, int(timesteps / 6))]])
	# imgs = [forward.forward_process(torch.tensor(colored[0]), torch.tensor([t])).numpy()[0,0] for t in [i for i in range(0,timesteps, int(timesteps / 6))]]
	# print(imgs)
	print(colored[0].shape)
	count = 6
	fig, axs = plt.subplots(1,count)
	for i in range(count):
		img = forward.forward_process(torch.tensor(colored[1]), torch.tensor([int(i * timesteps / count)])).numpy()
		# img = forward.forward_process(torch.tensor(colored[0]), torch.tensor([i * timesteps / count])).numpy()[0,0]
		print("result",img)
		axs[i].imshow(img)
		axs[i].axis("off")
		# axs[i].tight_layout()
	plt.tight_layout()
	plt.show()
