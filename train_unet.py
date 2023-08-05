from forward_process import Forward
from Unet import Unet
from torch.optim import Adam
import torch.nn.functional as F
import beta_schedule
import load_data
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sys
import time
from torch.optim.lr_scheduler import StepLR, LinearLR
from graph_loss import best_fit

class DiffusionModel(object):
	"""
		This class will handle the training and forward process for the diffusion model.

	Parameters
	----------
		beta_method : (int) -> array-like
			For the forward process
		timesteps : int
			For the forward process
		dataloader : dataset
			From the dataset library to load in data in batches
	"""

	def __init__(self, beta_schedule, dataloader, testloader = None, timesteps=300, epochs=10, learning_rate=5e-4):
		image_size = 28
		channels = 1
		self.timesteps = timesteps

		## Initialize the forward process ##
		self.forward_process = Forward(beta_schedule, timesteps+1)

		## Initialize the unet model ##
		self.device = torch.device('cuda:0')
		self.model = Unet(
		    dim=image_size,
		    channels=channels,
		    dim_mults=(1, 2, 4)
		)
		self.model.to(self.device)
		self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

		self.epochs = epochs
		self.dataloader = dataloader
		self.testloader = testloader

		self.train_loss = []
		self.test_loss = []

		# self.scheduler = LinearLR(self.optimizer,start_factor=1, end_factor=1/5, total_iters=epochs) 
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=1e-4)

	def get_test_loss(self, batch):
		self.optimizer.zero_grad()

		batch_size = batch["pixel_values"].shape[0]
		batch = batch["pixel_values"].to(self.device)

		# Algorithm 1 line 3: sample t uniformally for every example in the batch
		t = torch.randint(1, self.timesteps+1, (batch_size,), device=self.device).long()

		loss = self.loss(batch, t, "l1")

		return loss.item()

	def loss(self, x_0, t, loss_type, noise=None):
		if noise is None:
			noise = torch.randn_like(x_0)

		x_noisy_t_minus_1 = self.forward_process.forward_process(x_0, t-1, noise=noise)
		# x_noisy_t = self.forward_process.get_next_step(x_noisy_t_minus_1,t,noise = noise)
		x_noisy_t = self.forward_process.forward_process(x_0, t, noise=noise)

		predicted_x_t_minus_1 = self.model(x_noisy_t, t)

		#Use pytorch to calculate loss
		loss = self.get_loss_function(loss_type)(predicted_x_t_minus_1, x_noisy_t_minus_1)

		return loss

	def get_loss_function(self, loss_type):
		if loss_type == 'l1':
			return F.l1_loss
		elif loss_type == 'l2':
			return F.mse_loss
		elif loss_type == "huber":
			return F.smooth_l1_loss
		else:
			raise NotImplementedError()

	def train_once(self, step, batch, save = True):
		'''
			Trains the model over the entire training data one time
		'''
		self.optimizer.zero_grad()

		batch_size = batch["pixel_values"].shape[0]
		batch = batch["pixel_values"].to(self.device)

		# Algorithm 1 line 3: sample t uniformally for every example in the batch
		t = torch.randint(1, self.timesteps+1, (batch_size,), device=self.device).long()

		loss = self.loss(batch, t, "l1")

		if step % 10 == 0 and save:
			# print("Loss:", loss.item())
			self.train_loss.append(loss.item())

		loss.backward()
		self.optimizer.step()
		# Use the scheduler
		# self.scheduler.step()

	def train(self):
		'''
			Trains the model for the given number of epochs
		'''
		#initial test loss
		# if (self.testloader):
		# 	total = 0
		# 	times = 0
		# 	for batch in tqdm(self.testloader):
		# 		total += self.get_test_loss(batch)
		# 		times += 1
		# 	self.test_loss.append(total / times)

		for epoch in tqdm(range(self.epochs)):
			for step, batch in tqdm(enumerate(self.dataloader)):
				self.train_once(step, batch, epoch > 0)

			# Calculate test loss
			if (self.testloader and epoch > 0):
				total = 0
				times = 0
				for batch in tqdm(self.testloader):
					total += self.get_test_loss(batch)
					times += 1
				self.test_loss.append(total / times)

	def save(self, filename):
		print("Saved parameters to:", filename + ".t7")
		state = {
        	'net':self.model.state_dict()
		}
		torch.save(state, filename + '.t7')

	def display_loss_graph(self, fp=None):
		print(len(self.test_loss), len(self.train_loss))
		xtest = [1 + i for i in range(1, len(self.test_loss) + 1)]
		xtrain = [1 + i / len(self.train_loss) * (self.epochs-1) for i in range(len(self.train_loss))]

		plt.plot(xtrain, self.train_loss, label="training loss")
		plt.plot(xtest, self.test_loss, label="test loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		if (fp):
			plt.savefig(fp + ".png")
		plt.legend()
		plt.show()

	def display_training_loss(self, fp=None):
		xtrain = [i / len(self.train_loss) * self.epochs for i in range(len(self.train_loss))]

		plt.plot(xtrain, self.train_loss, label="training loss")
		plt.title("Training Loss over time")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		if (fp):
			plt.savefig(fp + ".png")
		plt.show()

	def display_test_loss(self, fp=None):
		xtest = [i for i in range(len(self.test_loss))]

		plt.plot(xtest, self.test_loss, label="test loss")
		plt.title("Test Loss over time")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend()
		if (fp):
			plt.savefig(fp + ".png")
		plt.show()

	def save_loss(self, fp):

		with open(fp+".txt", "w") as f:
			f.write("training\n")
			for i in self.train_loss:
				f.write(str(i))
				f.write("\n")
			f.write("testing\n")
			for i in self.test_loss:
				f.write(str(i))
				f.write("\n")
			

if __name__ == '__main__':
	epochs = 20 #int(sys.argv[1])
	learning_rate = 4e-4#float(sys.argv[2])

	model = DiffusionModel(beta_schedule.sigmoid_beta_schedule, load_data.get_dataloader(), load_data.get_tester(), epochs=epochs, learning_rate=learning_rate)
	model.train()
	# model.save("saves/" + sys.argv[1])
	name = time.ctime().replace(" ", "_").replace(":","-") + '_learingRate_' + str(learning_rate) + "_epochs_" + str(model.epochs)
	model.save("saves/" + name)
	model.display_loss_graph("loss/both_"+name)
	model.display_training_loss("loss/train_"+name)
	model.display_test_loss("loss/test_"+name)

	model.save_loss("loss/nums_"+name)

	best_fit(model.train_loss, epochs, 5)
