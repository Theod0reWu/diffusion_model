from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
# from datasets import load_dataset
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import numpy as np

transform = Compose([
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Lambda(lambda t: (t * 2) - 1)
		])

def transforms_data(examples):
	examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
	del examples["image"]

	return examples

def get_dataloader(only = -1, batch_size=64):
	dataset = load_dataset("mnist")

	if (only != -1):
		dataset = dataset.filter(lambda a : a['label'] == only)

	transformed_dataset = dataset.with_transform(transforms_data).remove_columns("label")

	return DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

def get_dataloader_torch(batch_size = 32):
	# Transforms images to a PyTorch Tensor
	tensor_transform = transforms.ToTensor()
	 
	# Download the MNIST Dataset
	dataset = datasets.MNIST(root = "./data",
	                         train = True,
	                         download = True,
	                         transform = tensor_transform)
	 
	# DataLoader is used to load the dataset
	# for training
	loader = torch.utils.data.DataLoader(dataset = dataset,
	                                     batch_size = batch_size,
	                                     shuffle = True)
	return loader

def get_tester(only = -1, batch_size = 64):
	dataset = load_dataset("mnist")

	if (only != -1):
		dataset = dataset.filter(lambda a : a['label'] == only)

	transformed_dataset = dataset.with_transform(transforms_data).remove_columns("label")

	return DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=True)

#### Generate colored images from mnist ################
def get_mnist_batch(batch, change_colors=False):
	import scipy
	from PIL import Image

	batch_size = batch.shape[0]

	lena = Image.open('resources/lena.png')
    
	# Select random batch (WxHxC)
	batch_raw = batch.reshape((batch_size, 28, 28, 1))
	
	# Resize
	batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in batch_raw])
	
	# Extend to RGB
	batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)
	
	# Make binary
	batch_binary = (batch_rgb > 0.5)
	# print(batch_binary.shape)
	
	batch = np.zeros((batch_size, 64, 64, 3))
	
	for i in range(batch_size):
	    # Take a random crop of the Lena image (background)
	    x_c = np.random.randint(0, lena.size[0] - 64)
	    y_c = np.random.randint(0, lena.size[1] - 64)
	    image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
	    image = np.asarray(image) / 255.0
	    if change_colors:
	        # Change color distribution
	        for j in range(3):
	            image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0
	    # Invert the colors at the location of the number
	    # print(image.shape)
	    image[batch_binary[i]] = 1 - image[batch_binary[i]]
	    
	    batch[i] = image
	
	# Map the whole batch to [-1, 1]
	#batch = batch / 0.5 	
	return batch

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	loader = get_dataloader()
	batch = next(iter(loader))['pixel_values']

	colored = get_mnist_batch(batch)

	count = 30
	plt.figure(figsize=(15,3))
	for i in range(count):
	    plt.subplot(2, count // 2, i+1)
	    plt.imshow(colored[i])
	    plt.axis('off')
	    
	plt.tight_layout()
	plt.show()